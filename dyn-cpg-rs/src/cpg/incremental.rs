use super::{
    Cpg, CpgError, NodeId,
    edge::{Edge, EdgeType},
    node::NodeType,
};
use crate::{
    cpg::spatial_index::SpatialIndex,
    diff::SourceEdit,
    languages::{cf_pass, data_dep_pass},
};
use std::{
    cmp::{max, min},
    collections::HashMap,
};
use tracing::{debug, warn};

impl Cpg {
    /// Incrementally update the CPG from the CST edits
    pub fn incremental_update(
        &mut self,
        edits: Vec<SourceEdit>,
        changed_ranges: impl ExactSizeIterator<Item = tree_sitter::Range>,
        new_tree: &tree_sitter::Tree,
    ) {
        debug!(
            "Incremental update with {} edits and {} changed ranges",
            edits.len(),
            changed_ranges.len()
        );

        let mut dirty_nodes = HashMap::new();
        for range in changed_ranges {
            debug!("TS Changed range: {:?}", range);
            if let Some(node_id) =
                self.get_smallest_node_id_containing_range(range.start_byte, range.end_byte)
            {
                dirty_nodes.insert(
                    node_id,
                    (
                        range.start_byte,
                        range.end_byte,
                        range.start_byte,
                        range.end_byte,
                    ),
                );
            } else {
                debug!(
                    "No node found for changed range: {:?}",
                    (range.start_byte, range.end_byte)
                );
            }
        }

        for edit in edits {
            debug!("Textual edit: {:?}", edit);
            if let Some(node_id) =
                self.get_smallest_node_id_containing_range(edit.old_start, edit.old_end)
            {
                dirty_nodes
                    .entry(node_id)
                    .and_modify(|existing_dirty| {
                        *existing_dirty = (
                            min(edit.old_start, existing_dirty.0),
                            max(edit.old_end, existing_dirty.1),
                            min(edit.new_start, existing_dirty.2),
                            max(edit.new_end, existing_dirty.3),
                        );
                    })
                    .or_insert((edit.old_start, edit.old_end, edit.new_start, edit.new_end));
            } else {
                debug!(
                    "No node found for edit range: {:?}",
                    (edit.old_start, edit.old_end)
                );
            }
        }

        debug!("Filtering {} dirty nodes", dirty_nodes.len());

        fn ranges_overlap(
            a: (usize, usize, usize, usize),
            b: (usize, usize, usize, usize),
        ) -> bool {
            // Check overlap in both old and new ranges
            let old_overlap = a.0 < b.1 && b.0 < a.1;
            let new_overlap = a.2 < b.3 && b.2 < a.3;
            old_overlap || new_overlap
        }

        fn merge_ranges(
            a: (usize, usize, usize, usize),
            b: (usize, usize, usize, usize),
        ) -> (usize, usize, usize, usize) {
            (
                min(a.0, b.0), // old_start
                max(a.1, b.1), // old_end
                min(a.2, b.2), // new_start
                max(a.3, b.3), // new_end
            )
        }

        // Group overlapping ranges
        let mut merged_ranges: Vec<(usize, usize, usize, usize)> = Vec::new();

        for (_, range) in dirty_nodes.iter() {
            let mut current_range = *range;
            let mut i = 0;

            // Try to merge with existing ranges
            while i < merged_ranges.len() {
                if ranges_overlap(current_range, merged_ranges[i]) {
                    // Merge ranges and remove the old one
                    current_range = merge_ranges(current_range, merged_ranges.remove(i));
                    // Don't increment i since we removed an element
                } else {
                    i += 1;
                }
            }

            merged_ranges.push(current_range);
        }

        debug!(
            "Merged {} overlapping ranges into {} ranges",
            dirty_nodes.len(),
            merged_ranges.len()
        );

        // For each merged range, find the appropriate node that contains it
        let dirty_nodes: Vec<(NodeId, (usize, usize, usize, usize))> = merged_ranges
            .into_iter()
            .filter_map(|range| {
                // Get all nodes that contain the merged range
                let candidates = self
                    .spatial_index
                    .get_nodes_covering_range(range.0, range.1)
                    .into_iter()
                    .filter(|node_id| {
                        // Only consider nodes that fully contain the range
                        let node_range = self.spatial_index.get_node_span(*node_id);
                        if let Some((start, end)) = node_range {
                            start <= range.0 && range.1 <= end
                        } else {
                            false
                        }
                    })
                    .collect::<Vec<_>>();

                // Choose the most appropriate node to rehydrate:
                // 1. Prefer structural control flow nodes over content nodes
                // 2. Among structural nodes, choose the one that best represents the change
                // 3. Fall back to smallest containing node if no structural nodes
                let containing_node = candidates
                    .into_iter()
                    .filter_map(|node_id| {
                        let node = self.get_node_by_id(&node_id)?;
                        let node_range = self.spatial_index.get_node_span(node_id)?;
                        let range_size = node_range.1 - node_range.0;

                        // Assign priority weights - lower values = higher priority
                        let priority_weight = match &node.type_ {
                            // Highest priority: Control flow structures that can change significantly
                            NodeType::Branch { .. } => 1,
                            NodeType::Loop { .. } => 1,

                            // High priority: Major structural elements
                            NodeType::Function { .. } => 10,
                            NodeType::Statement => 20,
                            NodeType::Block => 30,

                            // Medium priority: Expression-level constructs
                            NodeType::Expression => 100,
                            NodeType::Call => 100,
                            NodeType::Return => 100,

                            // Low priority: Leaf nodes and language-specific constructs
                            NodeType::Identifier => 1000,
                            NodeType::Comment => 1000,
                            NodeType::Type => 500,
                            NodeType::LanguageImplementation(_) => 800,

                            // Default priority for other nodes
                            _ => 200,
                        };

                        // Combined weight: prioritize by type first, then by size
                        Some((node_id, priority_weight + range_size))
                    })
                    .min_by_key(|(_, weight)| *weight)
                    .map(|(node_id, _)| node_id);

                if let Some(node_id) = containing_node {
                    debug!(
                        "Found containing node {:?} for merged range {:?}",
                        node_id, range
                    );
                    Some((node_id, range))
                } else {
                    debug!("No containing node found for merged range {:?}", range);
                    None
                }
            })
            .collect();

        debug!(
            "Rehydrating {} dirty nodes: {:?}",
            dirty_nodes.len(),
            dirty_nodes
        );
        let mut rehydrated_nodes = Vec::new();

        // Rehydrate dirty nodes
        for (id, pos) in dirty_nodes {
            debug!(
                "Attempting to rehydrate dirty node {:?} with position {:?}",
                id, pos
            );

            // Debug: Log the dirty node info before removal
            if let Some(node) = self.get_node_by_id(&id) {
                let range = self.spatial_index.get_node_span(id);
                debug!("Dirty node type: {:?}, range: {:?}", node.type_, range);

                // Count children to understand node size
                let child_count = self
                    .get_outgoing_edges(id)
                    .iter()
                    .filter(|e| e.type_ == EdgeType::SyntaxChild)
                    .count();
                debug!("Dirty node has {} children", child_count);
            } else {
                warn!("Could not find dirty node {:?} in CPG", id);
            }

            let new_node = self.rehydrate(id, pos, new_tree);
            match new_node {
                Ok(new_id) => {
                    debug!(
                        "Successfully rehydrated node {:?} to new id {:?}",
                        id, new_id
                    );
                    rehydrated_nodes.push(new_id);
                }
                Err(e) => {
                    warn!("Failed to rehydrate node {:?}: {}", id, e);
                }
            }
        }

        debug!(
            "Computing control flow for {} rehydrated nodes",
            rehydrated_nodes.len()
        );
        for node_id in rehydrated_nodes.clone() {
            match cf_pass(self, node_id) {
                Ok(()) => debug!("Successfully computed control flow for node {:?}", node_id),
                Err(e) => warn!(
                    "Failed to recompute control flow for node {:?}: {}",
                    node_id, e
                ),
            }
        }

        debug!(
            "Computing program dependence for {} rehydrated nodes",
            rehydrated_nodes.len()
        );
        for node_id in rehydrated_nodes {
            match data_dep_pass(self, node_id) {
                Ok(()) => debug!(
                    "Successfully computed data dependence for node {:?}",
                    node_id
                ),
                Err(e) => warn!(
                    "Failed to recompute data dependence for node {:?}: {}",
                    node_id, e
                ),
            }
        }

        debug!("Incremental update complete");
    }

    fn rehydrate(
        &mut self,
        id: NodeId,
        pos: (usize, usize, usize, usize),
        new_tree: &tree_sitter::Tree,
    ) -> Result<NodeId, CpgError> {
        debug!("Starting rehydration of node {:?}", id);

        // Check if the node exists or if it has been removed in a previous update
        if !self.nodes.contains_key(id) {
            warn!("Node {:?} does not exist in CPG, cannot rehydrate", id);
            return Err(CpgError::MissingField(format!(
                "Node {:?} does not exist in CPG",
                id
            )));
        }

        let is_current_root = self.root == Some(id);
        debug!("Node is root: {}", is_current_root);

        // Capture edge information before removal
        let old_left_sibling = self
            .get_incoming_edges(id)
            .into_iter()
            .find(|e| e.type_ == EdgeType::SyntaxSibling)
            .map(|e| {
                debug!("Found left sibling edge: {:?} -> {:?}", e.from, e.to);
                (e.from, e.properties.clone())
            });

        let old_right_sibling = self
            .get_outgoing_edges(id)
            .into_iter()
            .find(|e| e.type_ == EdgeType::SyntaxSibling)
            .map(|e| {
                debug!("Found right sibling edge: {:?} -> {:?}", e.from, e.to);
                (e.to, e.properties.clone())
            });

        let old_parent = self
            .get_incoming_edges(id)
            .into_iter()
            .find(|e| e.type_ == EdgeType::SyntaxChild)
            .map(|e| {
                debug!("Found parent edge: {:?} -> {:?}", e.from, e.to);
                (e.from, e.properties.clone())
            });

        debug!(
            "Captured edges - parent: {:?}, left sibling: {:?}, right sibling: {:?}",
            old_parent.is_some(),
            old_left_sibling.is_some(),
            old_right_sibling.is_some()
        );

        // Remove the old subtree
        debug!("Removing subtree rooted at {:?}", id);
        self.remove_subtree(id).map_err(|e| {
            CpgError::ConversionError(format!("Failed to remove old subtree: {}", e))
        })?;

        // Find the corresponding subtree in the new tree
        debug!(
            "Looking for subtree in new tree at range ({}, {})",
            pos.2, pos.3
        );
        let new_subtree_node = new_tree
            .root_node()
            .descendant_for_byte_range(pos.2, pos.3)
            .ok_or_else(|| {
                CpgError::MissingField(format!(
                    "No subtree found for range {:?} in new tree",
                    (pos.2, pos.3)
                ))
            })?;

        debug!(
            "Found new subtree node: kind={}, range=({}, {})",
            new_subtree_node.kind(),
            new_subtree_node.start_byte(),
            new_subtree_node.end_byte()
        );

        let mut cursor = new_subtree_node.walk();

        // Translate the new subtree
        debug!("Translating new subtree");
        let new_subtree_root = crate::languages::translate(self, &mut cursor).map_err(|e| {
            CpgError::ConversionError(format!("Failed to translate new subtree: {}", e))
        })?;

        debug!(
            "Translation complete, new subtree root: {:?}",
            new_subtree_root
        );

        // Reconstruct edges
        if let Some((left_sibling_from, properties)) = old_left_sibling {
            debug!(
                "Reconnecting left sibling: {:?} -> {:?}",
                left_sibling_from, new_subtree_root
            );
            self.add_edge(Edge {
                from: left_sibling_from,
                to: new_subtree_root,
                type_: EdgeType::SyntaxSibling,
                properties,
            });
        }

        if let Some((right_sibling_to, properties)) = old_right_sibling {
            debug!(
                "Reconnecting right sibling: {:?} -> {:?}",
                new_subtree_root, right_sibling_to
            );
            self.add_edge(Edge {
                from: new_subtree_root,
                to: right_sibling_to,
                type_: EdgeType::SyntaxSibling,
                properties,
            });
        }

        if let Some((parent_from, properties)) = old_parent {
            debug!(
                "Reconnecting parent: {:?} -> {:?}",
                parent_from, new_subtree_root
            );
            self.add_edge(Edge {
                from: parent_from,
                to: new_subtree_root,
                type_: EdgeType::SyntaxChild,
                properties,
            });
        } else if is_current_root {
            debug!(
                "Rehydrating root node {:?}, setting new root to {:?}",
                id, new_subtree_root
            );
            self.set_root(new_subtree_root);
        } else {
            warn!(
                "No parent edge found for node {:?}, but it's not the root node - this may indicate a problem",
                id
            );
        }

        debug!(
            "Rehydration complete for {:?} -> {:?}",
            id, new_subtree_root
        );
        Ok(new_subtree_root)
    }

    /// Recursively removes a subtree from the CPG by its root node ID
    /// This function now properly cleans up edges associated with the removed nodes.
    pub fn remove_subtree(&mut self, root: NodeId) -> Result<(), CpgError> {
        // 1. Recursively remove child subtrees first
        // Collect edges to avoid borrowing issues
        let child_edges: Vec<_> = self
            .get_outgoing_edges(root)
            .into_iter()
            .filter(|e| e.type_ == EdgeType::SyntaxChild)
            .cloned() // Clone edges to avoid holding references
            .collect();

        for edge in child_edges {
            self.remove_subtree(edge.to)?;
        }

        // 2. Now remove the root node itself and its associated edges
        // Remove the node data and spatial index entry
        self.nodes.remove(root);
        self.spatial_index.delete(root);

        // 3. Crucially: Remove all edges connected to this node
        // First collect ALL edge IDs that reference this node from the main edges SlotMap
        let mut edges_to_remove = Vec::new();
        for (edge_id, edge) in self.edges.iter() {
            if edge.from == root || edge.to == root {
                edges_to_remove.push(edge_id);
            }
        }

        // Remove each edge and update adjacency lists
        for edge_id in edges_to_remove {
            if let Some(edge) = self.edges.remove(edge_id) {
                // Remove from outgoing list of the 'from' node
                if let Some(outgoing_list) = self.outgoing.get_mut(&edge.from) {
                    outgoing_list.retain(|&id| id != edge_id);
                    if outgoing_list.is_empty() {
                        self.outgoing.remove(&edge.from);
                    }
                }

                // Remove from incoming list of the 'to' node
                if let Some(incoming_list) = self.incoming.get_mut(&edge.to) {
                    incoming_list.retain(|&id| id != edge_id);
                    if incoming_list.is_empty() {
                        self.incoming.remove(&edge.to);
                    }
                }
            }
        }

        // 4. Finally, remove any empty adjacency lists for the removed node
        self.incoming.remove(&root);
        self.outgoing.remove(&root);

        Ok(())
    }
}
