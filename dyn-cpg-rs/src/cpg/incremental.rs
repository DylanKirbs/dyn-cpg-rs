use super::{Cpg, CpgError, NodeId, edge::EdgeType, node::NodeType};
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
        new_source: Vec<u8>,
    ) {
        debug!(
            "[INCREMENTAL UPDATE] Update with {} edits and {} changed ranges",
            edits.len(),
            changed_ranges.len()
        );

        self.set_source(new_source);

        let mut dirty_nodes = HashMap::new();
        for range in changed_ranges {
            debug!("[INCREMENTAL UPDATE] TS Changed range: {:?}", range);
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
                    "[INCREMENTAL UPDATE] No node found for changed range: {:?}",
                    (range.start_byte, range.end_byte)
                );
            }
        }

        for edit in edits {
            debug!("[INCREMENTAL UPDATE] Textual edit: {:?}", edit);
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
                    "[INCREMENTAL UPDATE] No node found for edit range: {:?}",
                    (edit.old_start, edit.old_end)
                );
            }
        }

        debug!(
            "[INCREMENTAL UPDATE] Filtering {} dirty nodes",
            dirty_nodes.len()
        );

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
            "[INCREMENTAL UPDATE] Merged {} overlapping ranges into {} ranges",
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
                            NodeType::Identifier { .. } => 1000,
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
                        "[INCREMENTAL UPDATE] Found containing node {:?} for merged range {:?}",
                        node_id, range
                    );
                    Some((node_id, range))
                } else {
                    debug!(
                        "[INCREMENTAL UPDATE] No containing node found for merged range {:?}",
                        range
                    );
                    None
                }
            })
            .collect();

        debug!(
            "[INCREMENTAL UPDATE] Updating {} dirty nodes: {:?}",
            dirty_nodes.len(),
            dirty_nodes
        );

        // To avoid borrow checker issues, collect update info first
        let mut update_plan = Vec::new();
        for (id, _pos) in &dirty_nodes {
            let node_span = self.spatial_index.get_node_span(*id);
            if let Some((start, end)) = node_span {
                if let Some(cst_node) = new_tree.root_node().descendant_for_byte_range(start, end) {
                    update_plan.push((*id, cst_node));
                } else {
                    warn!(
                        "[INCREMENTAL UPDATE] No CST node found for CPG node {:?} in new tree",
                        id
                    );
                }
            } else {
                warn!("[INCREMENTAL UPDATE] No span found for CPG node {:?}", id);
            }
        }

        let mut updated_structures = Vec::new();
        let mut updated_functions = Vec::new();
        for (id, cst_node) in &update_plan {
            self.update_in_place_pairwise(*id, cst_node);
            let structure = crate::languages::get_container_parent(self, *id);
            if !updated_structures.contains(&structure) {
                updated_structures.push(structure);
            }
            if let Some(function) = crate::languages::get_containing_function(self, *id) {
                if !updated_functions.contains(&function) {
                    updated_functions.push(function);
                }
            }
        }

        debug!(
            "[INCREMENTAL UPDATE] Recomputing control flow for updated structures: {:?}",
            updated_structures
        );
        for structure in &updated_structures {
            if let Err(e) = cf_pass(self, *structure) {
                warn!(
                    "[INCREMENTAL UPDATE] Failed to recompute control flow for node {:?}: {}",
                    structure, e
                );
            }
        }
        debug!(
            "[INCREMENTAL UPDATE] Recomputing data dependence for updated functions: {:?}",
            updated_functions
        );
        for function in &updated_functions {
            if let Err(e) = data_dep_pass(self, *function) {
                warn!(
                    "[INCREMENTAL UPDATE] Failed to recompute data dependence for node {:?}: {}",
                    function, e
                );
            }
        }
        debug!("[INCREMENTAL UPDATE] Update complete");
    }

    /// Pairwise walk of CPG and CST, updating only changed nodes in place.
    /// This assumes the CPG and CST are structurally similar except for edits.
    pub fn update_in_place_pairwise(&mut self, cpg_node: NodeId, cst_node: &tree_sitter::Node) {
        let lang = self.get_language().clone();

        // Update the node type and span if the CST kind or span has changed
        let cpg_node_type = self.get_node_by_id(&cpg_node).map(|n| n.type_.clone());
        let cst_kind = cst_node.kind();
        let new_type = lang.map_node_kind(cst_kind);
        let cst_start = cst_node.start_byte();
        let cst_end = cst_node.end_byte();
        if let Some(old_type) = cpg_node_type {
            if old_type != new_type {
                if let Some(node) = self.get_node_by_id_mut(&cpg_node) {
                    node.type_ = new_type;
                }
            }
        }

        self.spatial_index.edit(cpg_node, cst_start, cst_end);
        // Update node properties (e.g., raw_kind)
        if let Some(node) = self.get_node_by_id_mut(&cpg_node) {
            node.properties
                .insert("raw_kind".to_string(), cst_kind.to_string());
        }

        // Update children with smarter matching (by kind and span)
        let cpg_children = self.ordered_syntax_children(cpg_node);
        let mut cst_cursor = cst_node.walk();
        let mut cst_children = Vec::new();
        if cst_cursor.goto_first_child() {
            loop {
                cst_children.push(cst_cursor.node());
                if !cst_cursor.goto_next_sibling() {
                    break;
                }
            }
        }

        // Build a list of (index, kind, span) for both CPG and CST children
        #[derive(Debug, Clone)]
        struct ChildInfo {
            kind: String,
            start: usize,
            end: usize,
        }
        let cpg_child_info: Vec<_> = cpg_children
            .iter()
            .map(|&id| {
                let node = self.get_node_by_id(&id).unwrap();
                let (start, end) = self.spatial_index.get_node_span(id).unwrap_or((0, 0));
                ChildInfo {
                    kind: node
                        .properties
                        .get("raw_kind")
                        .cloned()
                        .unwrap_or_else(|| node.type_.to_string()),
                    start,
                    end,
                }
            })
            .collect();
        let cst_child_info: Vec<_> = cst_children
            .iter()
            .map(|n| ChildInfo {
                kind: n.kind().to_string(),
                start: n.start_byte(),
                end: n.end_byte(),
            })
            .collect();

        // Greedy matching: for each CST child, try to find a CPG child with the same kind and overlapping span
        let mut matched_cpg = vec![false; cpg_children.len()];
        let mut matched_cst = vec![false; cst_children.len()];
        let mut pairs = Vec::new();
        for (cst_idx, cst) in cst_child_info.iter().enumerate() {
            let mut best: Option<(usize, usize)> = None;
            for (cpg_idx, cpg) in cpg_child_info.iter().enumerate() {
                if matched_cpg[cpg_idx] {
                    continue;
                }
                if cpg.kind == cst.kind {
                    // Prefer exact span match, otherwise any overlap
                    let overlap = cpg.start < cst.end && cst.start < cpg.end;
                    let exact = cpg.start == cst.start && cpg.end == cst.end;
                    if exact {
                        best = Some((cpg_idx, 2));
                        break;
                    } else if overlap {
                        best = Some((cpg_idx, 1));
                    } else if best.is_none() {
                        best = Some((cpg_idx, 0));
                    }
                }
            }
            if let Some((cpg_idx, _score)) = best {
                matched_cpg[cpg_idx] = true;
                matched_cst[cst_idx] = true;
                pairs.push((cpg_idx, cst_idx));
            }
        }

        for (cpg_idx, cst_idx) in pairs.iter().cloned() {
            let cpg_id = cpg_children[cpg_idx];
            let cst_child = &cst_children[cst_idx];
            self.update_in_place_pairwise(cpg_id, cst_child);
        }
        // Add new CST children that weren't matched
        for (cst_idx, was_matched) in matched_cst.iter().enumerate() {
            if !was_matched {
                let child = &cst_children[cst_idx];
                let type_ = lang.map_node_kind(child.kind());
                let mut node = crate::cpg::Node {
                    type_,
                    properties: std::collections::HashMap::new(),
                };
                node.properties
                    .insert("raw_kind".to_string(), child.kind().to_string());
                let id = self.add_node(node, child.start_byte(), child.end_byte());
                self.add_edge(crate::cpg::Edge {
                    from: cpg_node,
                    to: id,
                    type_: crate::cpg::EdgeType::SyntaxChild,
                    properties: std::collections::HashMap::new(),
                });
                self.update_in_place_pairwise(id, child);
            }
        }
        // Remove CPG children that weren't matched
        for (cpg_idx, was_matched) in matched_cpg.iter().enumerate() {
            if !was_matched {
                self.remove_subtree(cpg_children[cpg_idx]).ok();
            }
        }
    }

    /// Recursively removes a subtree from the CPG by its root node ID
    /// This function now properly cleans up edges associated with the removed nodes.
    pub fn remove_subtree(&mut self, root: NodeId) -> Result<(), CpgError> {
        let child_edges: Vec<_> = self
            .get_outgoing_edges(root)
            .into_iter()
            .filter(|e| e.type_ == EdgeType::SyntaxChild)
            .cloned()
            .collect();

        for edge in child_edges {
            self.remove_subtree(edge.to)?;
        }

        self.nodes.remove(root);
        self.spatial_index.delete(root);

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

        self.incoming.remove(&root);
        self.outgoing.remove(&root);

        Ok(())
    }
}
