use super::{Cpg, CpgError, NodeId, edge::EdgeType, node::NodeType};
use crate::{
    cpg::spatial_index::SpatialIndex,
    diff::SourceEdit,
    languages::{cf_pass, data_dep_pass, get_container_parent, translate},
};
use std::{
    cmp::{max, min},
    collections::HashMap,
};
use tracing::{debug, warn};

/// Configuration for the update strategy decision framework
#[derive(Debug, Clone)]
pub struct UpdateConfig {
    /// If more than this percentage of children change, rebuild the subtree
    pub rebuild_threshold_percentage: f32,
    /// If subtree depth exceeds this, rebuild instead of surgical update
    pub max_surgical_depth: usize,
    /// If more than this many structural operations, rebuild
    pub max_structural_operations: usize,
    /// Enable detailed logging for debugging
    pub debug_logging: bool,
}

impl Default for UpdateConfig {
    fn default() -> Self {
        Self {
            rebuild_threshold_percentage: 0.5, // 50%
            max_surgical_depth: 10,
            max_structural_operations: 20,
            debug_logging: false,
        }
    }
}

/// Edit operations for transforming CPG children to match CST children
#[derive(Debug, Clone)]
enum EditOperation {
    Insert { cst_index: usize, position: usize },
    Delete { cpg_index: usize },
    Modify { cpg_index: usize, cst_index: usize },
}

/// Information about a child node for matching
#[derive(Debug, Clone)]
struct ChildInfo {
    kind: String,
    start: usize,
    end: usize,
}

/// Metrics for decision making
#[derive(Debug)]
struct UpdateMetrics {
    children_changed_percentage: f32,
    subtree_depth: usize,
    structural_operations: usize,
    has_node_type_changes: bool,
}

impl Cpg {
    /// Incrementally update the CPG from the CST edits using the new systematic approach
    pub fn incremental_update(
        &mut self,
        edits: Vec<SourceEdit>,
        changed_ranges: impl ExactSizeIterator<Item = tree_sitter::Range>,
        new_tree: &tree_sitter::Tree,
        new_source: Vec<u8>,
    ) {
        self.incremental_update_with_config(
            edits,
            changed_ranges,
            new_tree,
            new_source,
            UpdateConfig::default(),
        )
    }

    /// Incrementally update with custom configuration
    pub fn incremental_update_with_config(
        &mut self,
        edits: Vec<SourceEdit>,
        changed_ranges: impl ExactSizeIterator<Item = tree_sitter::Range>,
        new_tree: &tree_sitter::Tree,
        new_source: Vec<u8>,
        config: UpdateConfig,
    ) {
        debug!(
            "[INCREMENTAL UPDATE] Starting incremental update with {} edits and {} changed ranges",
            edits.len(),
            changed_ranges.len()
        );

        // Update the CPG's source to match the new tree
        self.set_source(new_source);

        let mut dirty_nodes = HashMap::new();

        // Collect changed ranges from tree-sitter
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

        // Collect ranges from textual edits
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

        // Merge overlapping ranges
        let merged_ranges = self.merge_overlapping_ranges(dirty_nodes, &config);

        // Find appropriate nodes to update
        let dirty_nodes = self.find_update_nodes(merged_ranges, &config);

        debug!(
            "[INCREMENTAL UPDATE] Updating {} dirty nodes: {:?}",
            dirty_nodes.len(),
            dirty_nodes
        );

        // Phase 1: Clean slate - remove analysis edges
        for (node_id, _) in &dirty_nodes {
            self.clean_analysis_edges_recursive(*node_id);
        }

        // Phase 2 & 3: Update nodes using strategic approach
        let mut updated_structures = Vec::new();
        let mut updated_functions = Vec::new();

        // First, collect all container structures and functions that might be affected
        for (node_id, _range) in &dirty_nodes {
            // Track structures and functions BEFORE updating (in case nodes get removed)
            let structure = crate::languages::get_container_parent(self, *node_id);
            if !updated_structures.contains(&structure) {
                updated_structures.push(structure);
                debug!(
                    "[INCREMENTAL UPDATE] Tracking structure for recomputation: {:?}",
                    structure
                );
            }
            if let Some(function) = crate::languages::get_containing_function(self, *node_id) {
                if !updated_functions.contains(&function) {
                    updated_functions.push(function);
                    debug!(
                        "[INCREMENTAL UPDATE] Tracking function for recomputation: {:?}",
                        function
                    );
                }
            }
        }

        // Now perform the actual updates
        for (node_id, _range) in &dirty_nodes {
            let node_span = self.spatial_index.get_node_span(*node_id);
            if let Some((start, end)) = node_span {
                if let Some(cst_node) = new_tree.root_node().descendant_for_byte_range(start, end) {
                    self.strategic_update(*node_id, &cst_node, &config);
                } else {
                    warn!(
                        "[INCREMENTAL UPDATE] No CST node found for CPG node {:?} in new tree",
                        node_id
                    );
                }
            } else {
                warn!(
                    "[INCREMENTAL UPDATE] No span found for CPG node {:?}",
                    node_id
                );
            }
        }

        // Phase 4: Recompute analysis passes
        // Instead of tracking node IDs that might change, use spatial queries
        // to find current nodes in the affected byte ranges
        debug!("[INCREMENTAL UPDATE] Recomputing analysis passes for affected byte ranges");

        let mut all_structures = std::collections::HashSet::new();
        let mut all_functions = std::collections::HashSet::new();

        // Collect all affected byte ranges
        let mut affected_ranges = Vec::new();
        for (_node_id, (_old_start, _old_end, new_start, new_end)) in &dirty_nodes {
            // Use the new range since that's what exists now
            affected_ranges.push((*new_start, *new_end));
        }

        // Find all current nodes in the affected ranges
        for (start, end) in affected_ranges {
            // Get all nodes that overlap with this range - use covering_range to include boundary nodes
            let overlapping_nodes = self.spatial_index.get_nodes_covering_range(start, end);

            for node_id in overlapping_nodes {
                // Find containing structure and function for each overlapping node
                let structure = get_container_parent(self, node_id);
                all_structures.insert(structure);

                if let Some(function) = crate::languages::get_containing_function(self, node_id) {
                    all_functions.insert(function);
                }
            }
        }

        debug!(
            "[INCREMENTAL UPDATE] Found {} structures and {} functions to recompute after spatial query",
            all_structures.len(),
            all_functions.len()
        );

        for structure in &all_structures {
            debug!(
                "[INCREMENTAL UPDATE] Running CF pass on structure {:?}",
                structure
            );

            // Debug: Check the children of this node
            let children = self.get_outgoing_edges(*structure);
            debug!(
                "[INCREMENTAL UPDATE] Structure {:?} has {} outgoing edges",
                structure,
                children.len()
            );
            for edge in children.iter().take(5) {
                debug!(
                    "[INCREMENTAL UPDATE]   Edge: {:?} -> {:?} ({:?})",
                    edge.from, edge.to, edge.type_
                );
            }

            if let Err(e) = cf_pass(self, *structure) {
                debug!(
                    "[INCREMENTAL UPDATE] Failed to recompute control flow for node {:?}: {}",
                    structure, e
                );
            } else {
                debug!(
                    "[INCREMENTAL UPDATE] Successfully recomputed CF for structure {:?}",
                    structure
                );
            }
        }

        for function in &all_functions {
            debug!(
                "[INCREMENTAL UPDATE] Running DD pass on function {:?}",
                function
            );

            if let Err(e) = data_dep_pass(self, *function) {
                debug!(
                    "[INCREMENTAL UPDATE] Failed to recompute data dependence for node {:?}: {}",
                    function, e
                );
            } else {
                debug!(
                    "[INCREMENTAL UPDATE] Successfully recomputed DD for function {:?}",
                    function
                );
            }
        }

        debug!("[INCREMENTAL UPDATE] Update complete");
    }

    /// Merge overlapping dirty node ranges
    fn merge_overlapping_ranges(
        &self,
        dirty_nodes: HashMap<NodeId, (usize, usize, usize, usize)>,
        config: &UpdateConfig,
    ) -> Vec<(usize, usize, usize, usize)> {
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

        if config.debug_logging {
            debug!(
                "[INCREMENTAL UPDATE] Merged {} overlapping ranges into {} ranges",
                dirty_nodes.len(),
                merged_ranges.len()
            );
        }

        merged_ranges
    }

    /// Find appropriate nodes to update for each merged range
    fn find_update_nodes(
        &self,
        merged_ranges: Vec<(usize, usize, usize, usize)>,
        config: &UpdateConfig,
    ) -> Vec<(NodeId, (usize, usize, usize, usize))> {
        merged_ranges
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

                // Choose the most appropriate node to update
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
                    if config.debug_logging {
                        debug!(
                            "[INCREMENTAL UPDATE] Found containing node {:?} for merged range {:?}",
                            node_id, range
                        );
                    }
                    Some((node_id, range))
                } else {
                    if config.debug_logging {
                        debug!(
                            "[INCREMENTAL UPDATE] No containing node found for merged range {:?}",
                            range
                        );
                    }
                    None
                }
            })
            .collect()
    }

    /// Phase 1: Clean analysis edges recursively from a subtree
    fn clean_analysis_edges_recursive(&mut self, node_id: NodeId) {
        // Remove all control flow and data dependence edges from this node and descendants
        let descendants = self.post_dfs_ordered_syntax_descendants(node_id);

        for descendant in descendants {
            let mut edges_to_remove = Vec::new();

            // Collect all analysis edges (preserve syntax structure)
            for edge in self.get_outgoing_edges(descendant) {
                match edge.type_ {
                    EdgeType::ControlFlowFunctionReturn
                    | EdgeType::ControlFlowEpsilon
                    | EdgeType::ControlFlowTrue
                    | EdgeType::ControlFlowFalse
                    | EdgeType::PDControlTrue
                    | EdgeType::PDControlFalse
                    | EdgeType::PDData(_) => {
                        // Find the edge ID to remove
                        for (edge_id, stored_edge) in self.edges.iter() {
                            if stored_edge.from == edge.from
                                && stored_edge.to == edge.to
                                && stored_edge.type_ == edge.type_
                            {
                                edges_to_remove.push(edge_id);
                                break;
                            }
                        }
                    }
                    _ => {} // Keep syntax edges
                }
            }

            // Remove the collected edges
            for edge_id in edges_to_remove {
                self.remove_edge(edge_id);
            }
        }
    }

    /// Phase 2 & 3: Strategic update - decide between surgical update or full rebuild
    fn strategic_update(
        &mut self,
        cpg_node: NodeId,
        cst_node: &tree_sitter::Node,
        config: &UpdateConfig,
    ) {
        // Gather metrics for decision making
        let metrics = self.compute_update_metrics(cpg_node, cst_node, config);

        // Decision logic based on configured thresholds
        let should_rebuild = metrics.children_changed_percentage
            > config.rebuild_threshold_percentage
            || metrics.subtree_depth > config.max_surgical_depth
            || metrics.structural_operations > config.max_structural_operations
            || metrics.has_node_type_changes;

        if config.debug_logging {
            debug!(
                "[STRATEGIC UPDATE] Node {:?}: changed={}%, depth={}, ops={}, type_change={}, decision={}",
                cpg_node,
                metrics.children_changed_percentage * 100.0,
                metrics.subtree_depth,
                metrics.structural_operations,
                metrics.has_node_type_changes,
                if should_rebuild {
                    "REBUILD"
                } else {
                    "SURGICAL"
                }
            );
        }

        if should_rebuild {
            self.full_rebuild_subtree(cpg_node, cst_node);
        } else {
            self.surgical_update_subtree(cpg_node, cst_node, config);
        }
    }

    /// Compute metrics for update decision making
    fn compute_update_metrics(
        &self,
        cpg_node: NodeId,
        cst_node: &tree_sitter::Node,
        _config: &UpdateConfig,
    ) -> UpdateMetrics {
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

        // Check for node type changes
        let current_node = self.get_node_by_id(&cpg_node);
        let lang = self.get_language().clone();
        let new_type = lang.map_node_kind(cst_node.kind());
        let has_node_type_changes = current_node.map_or(true, |n| n.type_ != new_type);

        // Compute similarity between children
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

        // Compute edit operations to estimate complexity
        let edit_ops = self.compute_edit_operations(&cpg_child_info, &cst_child_info);
        let structural_operations = edit_ops.len();

        // Count how many children are being changed
        let changed_children = edit_ops
            .iter()
            .filter(|op| {
                matches!(
                    op,
                    EditOperation::Insert { .. } | EditOperation::Delete { .. }
                )
            })
            .count();
        let total_children = cpg_children.len().max(cst_children.len());
        let children_changed_percentage = if total_children > 0 {
            changed_children as f32 / total_children as f32
        } else {
            0.0
        };

        // Compute subtree depth
        let subtree_depth = self.compute_subtree_depth(cpg_node);

        UpdateMetrics {
            children_changed_percentage,
            subtree_depth,
            structural_operations,
            has_node_type_changes,
        }
    }

    /// Compute edit operations using longest common subsequence approach
    fn compute_edit_operations(
        &self,
        cpg_children: &[ChildInfo],
        cst_children: &[ChildInfo],
    ) -> Vec<EditOperation> {
        let mut operations = Vec::new();

        // Simple greedy matching for now - can be improved with proper LCS algorithm
        let mut matched_cpg = vec![false; cpg_children.len()];
        let mut matched_cst = vec![false; cst_children.len()];

        // First pass: exact matches
        for (cst_idx, cst_child) in cst_children.iter().enumerate() {
            for (cpg_idx, cpg_child) in cpg_children.iter().enumerate() {
                if matched_cpg[cpg_idx] || matched_cst[cst_idx] {
                    continue;
                }

                if cpg_child.kind == cst_child.kind
                    && cpg_child.start == cst_child.start
                    && cpg_child.end == cst_child.end
                {
                    matched_cpg[cpg_idx] = true;
                    matched_cst[cst_idx] = true;
                    operations.push(EditOperation::Modify {
                        cpg_index: cpg_idx,
                        cst_index: cst_idx,
                    });
                    break;
                }
            }
        }

        // Second pass: similar matches
        for (cst_idx, cst_child) in cst_children.iter().enumerate() {
            if matched_cst[cst_idx] {
                continue;
            }

            for (cpg_idx, cpg_child) in cpg_children.iter().enumerate() {
                if matched_cpg[cpg_idx] {
                    continue;
                }

                if cpg_child.kind == cst_child.kind {
                    matched_cpg[cpg_idx] = true;
                    matched_cst[cst_idx] = true;
                    operations.push(EditOperation::Modify {
                        cpg_index: cpg_idx,
                        cst_index: cst_idx,
                    });
                    break;
                }
            }
        }

        // Third pass: insertions and deletions
        for (cst_idx, _) in cst_children.iter().enumerate() {
            if !matched_cst[cst_idx] {
                operations.push(EditOperation::Insert {
                    cst_index: cst_idx,
                    position: cst_idx,
                });
            }
        }

        for (cpg_idx, _) in cpg_children.iter().enumerate() {
            if !matched_cpg[cpg_idx] {
                operations.push(EditOperation::Delete { cpg_index: cpg_idx });
            }
        }

        operations
    }

    /// Compute subtree depth for complexity estimation
    fn compute_subtree_depth(&self, root: NodeId) -> usize {
        let mut max_depth = 0;
        let mut stack = vec![(root, 0)];

        while let Some((node, depth)) = stack.pop() {
            max_depth = max_depth.max(depth);

            for child in self.ordered_syntax_children(node) {
                stack.push((child, depth + 1));
            }
        }

        max_depth
    }

    /// Phase 3A: Surgical update using sequence alignment
    fn surgical_update_subtree(
        &mut self,
        cpg_node: NodeId,
        cst_node: &tree_sitter::Node,
        config: &UpdateConfig,
    ) {
        if config.debug_logging {
            debug!("[SURGICAL UPDATE] Updating node {:?} surgically", cpg_node);
        }

        let lang = self.get_language().clone();

        // Update the node itself
        self.update_node_properties(cpg_node, cst_node);

        // Get children
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

        // Build child info
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

        // Compute edit operations
        let edit_ops = self.compute_edit_operations(&cpg_child_info, &cst_child_info);

        // Remove syntax sibling edges for affected children
        self.clear_sibling_edges(&cpg_children);

        // Apply operations in order: deletes, modifies, inserts
        let mut deletions: Vec<_> = edit_ops
            .iter()
            .filter_map(|op| {
                if let EditOperation::Delete { cpg_index } = op {
                    Some(*cpg_index)
                } else {
                    None
                }
            })
            .collect();
        deletions.sort_by(|a, b| b.cmp(a)); // Reverse order to avoid index shifting

        for cpg_index in deletions {
            if cpg_index < cpg_children.len() {
                self.remove_subtree(cpg_children[cpg_index]).ok();
            }
        }

        // Apply modifies
        for op in &edit_ops {
            if let EditOperation::Modify {
                cpg_index,
                cst_index,
            } = op
            {
                if *cpg_index < cpg_children.len() && *cst_index < cst_children.len() {
                    self.strategic_update(
                        cpg_children[*cpg_index],
                        &cst_children[*cst_index],
                        config,
                    );
                }
            }
        }

        // Apply inserts
        for op in &edit_ops {
            if let EditOperation::Insert {
                cst_index,
                position: _,
            } = op
            {
                if *cst_index < cst_children.len() {
                    let child = &cst_children[*cst_index];
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
                    self.strategic_update(id, child, config);
                }
            }
        }

        // Rebuild sibling chains
        let updated_children = self.ordered_syntax_children(cpg_node);
        self.rebuild_sibling_chain(&updated_children);
    }

    /// Phase 3B: Full rebuild of subtree
    fn full_rebuild_subtree(&mut self, cpg_node: NodeId, cst_node: &tree_sitter::Node) {
        debug!("[FULL REBUILD] Rebuilding subtree at node {:?}", cpg_node);

        // Save parent information before rebuilding
        let parent_edges: Vec<_> = self
            .get_incoming_edges(cpg_node)
            .into_iter()
            .filter(|e| e.type_ == EdgeType::SyntaxChild)
            .map(|e| e.from) // Just collect the parent node IDs
            .collect();

        // Remove the entire old subtree
        self.remove_subtree(cpg_node).ok();

        // Build new subtree using proper language translation
        let mut cursor = cst_node.walk();
        let new_subtree_root = match translate(self, &mut cursor) {
            Ok(root) => root,
            Err(e) => {
                warn!("[FULL REBUILD] Failed to translate CST subtree: {}", e);
                return;
            }
        };

        // Reconnect to parent nodes
        for parent_id in parent_edges {
            self.add_edge(crate::cpg::Edge {
                from: parent_id,
                to: new_subtree_root,
                type_: EdgeType::SyntaxChild,
                properties: std::collections::HashMap::new(),
            });
        }
    }

    /// Update node properties (type, span, etc.)
    fn update_node_properties(&mut self, cpg_node: NodeId, cst_node: &tree_sitter::Node) {
        let lang = self.get_language().clone();
        let cst_kind = cst_node.kind();
        let new_type = lang.map_node_kind(cst_kind);
        let cst_start = cst_node.start_byte();
        let cst_end = cst_node.end_byte();

        // Update node type if changed
        if let Some(node) = self.get_node_by_id_mut(&cpg_node) {
            node.type_ = new_type;
            node.properties
                .insert("raw_kind".to_string(), cst_kind.to_string());
        }

        // Update spatial index
        self.spatial_index.edit(cpg_node, cst_start, cst_end);
    }

    /// Clear syntax sibling edges for a set of children
    fn clear_sibling_edges(&mut self, children: &[NodeId]) {
        let mut edges_to_remove = Vec::new();

        for &child in children {
            for edge in self.get_outgoing_edges(child) {
                if edge.type_ == EdgeType::SyntaxSibling {
                    // Find the edge ID
                    for (edge_id, stored_edge) in self.edges.iter() {
                        if stored_edge.from == edge.from
                            && stored_edge.to == edge.to
                            && stored_edge.type_ == edge.type_
                        {
                            edges_to_remove.push(edge_id);
                            break;
                        }
                    }
                }
            }
        }

        for edge_id in edges_to_remove {
            self.remove_edge(edge_id);
        }
    }

    /// Rebuild sibling chain in left-to-right order
    fn rebuild_sibling_chain(&mut self, children: &[NodeId]) {
        for i in 0..(children.len().saturating_sub(1)) {
            self.add_edge(crate::cpg::Edge {
                from: children[i],
                to: children[i + 1],
                type_: EdgeType::SyntaxSibling,
                properties: std::collections::HashMap::new(),
            });
        }
    }

    /// DEPRECATED: This method has been replaced by the strategic update system.
    /// Use `strategic_update` instead for proper incremental updates.
    #[deprecated(note = "Use strategic_update instead")]
    pub fn update_in_place_pairwise(&mut self, cpg_node: NodeId, cst_node: &tree_sitter::Node) {
        // For backward compatibility, delegate to strategic update with default config
        let config = UpdateConfig::default();
        self.strategic_update(cpg_node, cst_node, &config);
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
