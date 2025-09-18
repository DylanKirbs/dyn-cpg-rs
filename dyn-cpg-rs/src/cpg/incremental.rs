use super::{Cpg, CpgError, NodeId, edge::EdgeType, node::NodeType};
use crate::{
    cpg::spatial_index::SpatialIndex,
    diff::SourceEdit,
    languages::{cf_pass, data_dep_pass, post_translate_node, pre_translate_node},
};
use std::{
    cmp::{max, min},
    collections::{HashMap, HashSet},
};
use tracing::{debug, trace, warn};

/// Configuration for incremental update thresholds
#[derive(Debug, Clone)]
pub struct UpdateThresholds {
    /// If more than this percentage of children change, rebuild instead of surgical update
    pub max_children_change_percentage: f32,
    /// If subtree depth exceeds this, rebuild instead of surgical update  
    pub max_surgical_depth: usize,
    /// If number of operations exceeds this, rebuild instead of surgical update
    pub max_operations_count: usize,
}

impl Default for UpdateThresholds {
    fn default() -> Self {
        Self {
            max_children_change_percentage: 1.0,
            max_surgical_depth: 6,
            max_operations_count: 8,
        }
    }
}
/// Metrics for deciding between surgical update vs full rebuild
#[derive(Debug)]
#[allow(dead_code)] // Allow unused fields for debugging
struct UpdateMetrics {
    children_change_percentage: f32,
    subtree_depth: usize,
    operations_count: usize,
    has_structural_type_changes: bool,
}

/// Edit operations for sequence alignment
#[derive(Debug, Clone)]
enum EditOperation {
    Insert { cst_index: usize },
    Delete { cpg_index: usize },
    Modify { cpg_index: usize, cst_index: usize },
}

impl Cpg {
    /// Calculate edit operations needed to transform CPG children to match CST children
    /// Uses Longest Common Subsequence (LCS) algorithm for optimal sequence alignment
    fn calculate_edit_operations(
        &self,
        cpg_children: &[NodeId],
        cst_children: &[tree_sitter::Node],
    ) -> Vec<EditOperation> {
        // Build similarity matrix for node matching
        // Nodes matched on type compatibility (70%) and span overlap (30%)
        let similarity_matrix = self.build_similarity_matrix(cpg_children, cst_children);

        // Compute LCS using dynamic programming
        let lcs_table = self.compute_lcs_table(&similarity_matrix);

        // Backtrack to find optimal edit sequence
        self.backtrack_edit_operations(&lcs_table, cpg_children, cst_children, &similarity_matrix)
    }

    /// Build similarity matrix between CPG nodes and CST nodes
    fn build_similarity_matrix(
        &self,
        cpg_children: &[NodeId],
        cst_children: &[tree_sitter::Node],
    ) -> Vec<Vec<f32>> {
        let mut matrix = vec![vec![0.0; cst_children.len()]; cpg_children.len()];

        for (cpg_idx, &cpg_node_id) in cpg_children.iter().enumerate() {
            if let Some(cpg_node) = self.get_node_by_id(&cpg_node_id) {
                for (cst_idx, cst_node) in cst_children.iter().enumerate() {
                    // Type compatibility score (70% weight)
                    let cst_type = self.language.map_node_kind(cst_node.kind());
                    let type_score = if cpg_node.type_ == cst_type { 1.0 } else { 0.0 };

                    // Span overlap score (30% weight)
                    let span_score = self.calculate_span_overlap(cpg_node_id, *cst_node);

                    // Combined similarity score
                    matrix[cpg_idx][cst_idx] = 0.7 * type_score + 0.3 * span_score;
                }
            }
        }

        matrix
    }

    /// Calculate span overlap between CPG node and CST node
    fn calculate_span_overlap(&self, cpg_node_id: NodeId, cst_node: tree_sitter::Node) -> f32 {
        let cpg_span = self.spatial_index.get_node_span(cpg_node_id);
        let cst_start = cst_node.start_byte();
        let cst_end = cst_node.end_byte();

        if let Some((cpg_start, cpg_end)) = cpg_span {
            if cpg_end <= cpg_start || cst_end <= cst_start {
                return 0.0;
            }

            let overlap_start = max(cpg_start, cst_start);
            let overlap_end = min(cpg_end, cst_end);

            if overlap_start >= overlap_end {
                0.0
            } else {
                let overlap_len = overlap_end - overlap_start;
                let total_len = max(cpg_end - cpg_start, cst_end - cst_start);
                overlap_len as f32 / total_len as f32
            }
        } else {
            // No span information available
            0.0
        }
    }

    /// Compute LCS table using dynamic programming
    fn compute_lcs_table(&self, similarity_matrix: &[Vec<f32>]) -> Vec<Vec<usize>> {
        let m = similarity_matrix.len();
        let n = if m > 0 { similarity_matrix[0].len() } else { 0 };
        let mut lcs = vec![vec![0; n + 1]; m + 1];

        // Threshold for considering nodes as matching (similarity >= 0.6)
        // Higher threshold for commit-level batch processing to avoid false matches
        let similarity_threshold = 0.6;

        for i in 1..=m {
            for j in 1..=n {
                if similarity_matrix[i - 1][j - 1] >= similarity_threshold {
                    lcs[i][j] = lcs[i - 1][j - 1] + 1;
                } else {
                    lcs[i][j] = max(lcs[i - 1][j], lcs[i][j - 1]);
                }
            }
        }

        lcs
    }

    /// Backtrack through LCS table to generate edit operations
    fn backtrack_edit_operations(
        &self,
        lcs_table: &[Vec<usize>],
        cpg_children: &[NodeId],
        cst_children: &[tree_sitter::Node],
        similarity_matrix: &[Vec<f32>],
    ) -> Vec<EditOperation> {
        let mut operations = Vec::new();
        let mut i = cpg_children.len();
        let mut j = cst_children.len();
        let similarity_threshold = 0.6;

        while i > 0 || j > 0 {
            if i > 0 && j > 0 && similarity_matrix[i - 1][j - 1] >= similarity_threshold {
                // Match found - only generate modify if there are actual differences
                let cpg_node_id = cpg_children[i - 1];
                let cst_node = cst_children[j - 1];

                if let Some(cpg_node) = self.get_node_by_id(&cpg_node_id) {
                    let cpg_kind = cpg_node
                        .properties
                        .get("raw_kind")
                        .cloned()
                        .unwrap_or_else(|| cpg_node.type_.to_string());
                    let cst_kind = cst_node.kind().to_string();

                    let current_span = self.spatial_index.get_node_span(cpg_node_id);
                    let new_span = (cst_node.start_byte(), cst_node.end_byte());

                    // Only modify if there are actual differences
                    if cpg_kind != cst_kind || current_span != Some(new_span) {
                        operations.push(EditOperation::Modify {
                            cpg_index: i - 1,
                            cst_index: j - 1,
                        });
                    }
                    // If nodes are actually identical, skip modify operation
                }
                i -= 1;
                j -= 1;
            } else if i > 0 && (j == 0 || lcs_table[i - 1][j] >= lcs_table[i][j - 1]) {
                // Delete from CPG
                operations.push(EditOperation::Delete { cpg_index: i - 1 });
                i -= 1;
            } else if j > 0 {
                // Insert into CPG
                operations.push(EditOperation::Insert { cst_index: j - 1 });
                j -= 1;
            }
        }

        // Reverse to get operations in forward order
        operations.reverse();
        operations
    }

    /// Decide whether to use surgical update or full rebuild
    fn should_rebuild(
        &self,
        cpg_node: NodeId,
        cst_node: &tree_sitter::Node,
        thresholds: &UpdateThresholds,
    ) -> (bool, UpdateMetrics) {
        let cpg_children = self.ordered_syntax_children(cpg_node);
        let mut cst_children = Vec::new();
        let mut cst_cursor = cst_node.walk();
        if cst_cursor.goto_first_child() {
            loop {
                cst_children.push(cst_cursor.node());
                if !cst_cursor.goto_next_sibling() {
                    break;
                }
            }
        }

        // Calculate metrics
        let operations_count = self
            .calculate_edit_operations(&cpg_children, &cst_children)
            .len();
        let children_change_percentage = if cpg_children.is_empty() {
            if cst_children.is_empty() { 0.0 } else { 1.0 }
        } else {
            operations_count as f32 / cpg_children.len() as f32
        };

        let subtree_depth = self.calculate_subtree_depth(cpg_node);

        // Check for structural type changes
        let cpg_node_type = self.get_node_by_id(&cpg_node).map(|n| &n.type_);
        let cst_kind = cst_node.kind();
        let new_type = self.language.map_node_kind(cst_kind);
        let has_structural_type_changes =
            cpg_node_type.map_or(false, |old_type| old_type != &new_type);

        let metrics = UpdateMetrics {
            children_change_percentage,
            subtree_depth,
            operations_count,
            has_structural_type_changes,
        };

        // Decision logic using weighted scoring
        let should_rebuild = children_change_percentage > thresholds.max_children_change_percentage
            || subtree_depth > thresholds.max_surgical_depth
            || operations_count > thresholds.max_operations_count
            || has_structural_type_changes;

        (should_rebuild, metrics)
    }

    /// Find a parent node that could contain a range that doesn't map to any existing node
    /// This is useful when tree-sitter reports changed ranges for newly added structure
    fn find_parent_node_for_range(&self, start_byte: usize, end_byte: usize) -> Option<NodeId> {
        // Find all nodes that could potentially contain this range
        // Look for nodes whose span starts before start_byte and could be extended to include end_byte
        let mut candidates = Vec::new();

        for (node_id, node) in &self.nodes {
            if let Some((node_start, node_end)) = self.spatial_index.get_node_span(node_id) {
                // Consider nodes that start before our range and could be parents
                if node_start <= start_byte {
                    // Calculate how much the node would need to be extended to contain the range
                    let extension_needed = if end_byte > node_end {
                        end_byte - node_end
                    } else {
                        0
                    };

                    candidates.push((node_id, node_start, node_end, extension_needed, &node.type_));
                }
            }
        }

        if candidates.is_empty() {
            return None;
        }

        // Sort candidates by:
        // 1. Prefer structural nodes (blocks, functions, statements)
        // 2. Prefer nodes that need less extension
        // 3. Prefer nodes that start closer to the range
        candidates.sort_by(|a, b| {
            let (_, start_a, _, ext_a, type_a) = a;
            let (_, start_b, _, ext_b, type_b) = b;

            // Assign priority weights for node types
            let priority_a = match type_a {
                NodeType::Block => 1,
                NodeType::Branch { .. } => 2,
                NodeType::Function { .. } => 3,
                NodeType::Statement => 4,
                _ => 10,
            };

            let priority_b = match type_b {
                NodeType::Block => 1,
                NodeType::Branch { .. } => 2,
                NodeType::Function { .. } => 3,
                NodeType::Statement => 4,
                _ => 10,
            };

            // Compare by priority first
            match priority_a.cmp(&priority_b) {
                std::cmp::Ordering::Equal => {
                    // If same priority, prefer less extension needed
                    match ext_a.cmp(ext_b) {
                        std::cmp::Ordering::Equal => {
                            // If same extension, prefer closer start
                            (start_byte - start_a).cmp(&(start_byte - start_b))
                        }
                        other => other,
                    }
                }
                other => other,
            }
        });

        candidates.first().map(|(node_id, _, _, _, _)| *node_id)
    }

    /// Calculate the depth of a subtree
    fn calculate_subtree_depth(&self, root: NodeId) -> usize {
        let children = self.ordered_syntax_children(root);
        if children.is_empty() {
            1
        } else {
            1 + children
                .iter()
                .map(|&child| self.calculate_subtree_depth(child))
                .max()
                .unwrap_or(0)
        }
    }

    /// Remove all analysis edges (control flow and data dependence) from a subtree
    /// while preserving structural edges (SyntaxChild and SyntaxSibling)
    fn clean_analysis_edges(&mut self, root: NodeId) {
        let mut to_remove = Vec::new();

        // Collect all analysis edges in the subtree
        fn collect_analysis_edges_recursive(
            cpg: &Cpg,
            node: NodeId,
            edges_to_remove: &mut Vec<super::EdgeId>,
        ) {
            // Check outgoing edges from this node
            if let Some(outgoing_edge_ids) = cpg.outgoing.get(&node) {
                for &edge_id in outgoing_edge_ids {
                    if let Some(edge) = cpg.edges.get(edge_id) {
                        match edge.type_ {
                            EdgeType::ControlFlowTrue
                            | EdgeType::ControlFlowFalse
                            | EdgeType::ControlFlowEpsilon
                            | EdgeType::ControlFlowFunctionReturn
                            | EdgeType::PDControlTrue
                            | EdgeType::PDControlFalse
                            | EdgeType::PDData(_) => {
                                trace!(
                                    "[CLEAN ANALYSIS EDGES] [COLLECT] Marking edge {:?} for removal: {:?} -> {:?} type: {:?}",
                                    edge_id, edge.from, edge.to, edge.type_
                                );
                                edges_to_remove.push(edge_id);
                            }
                            _ => {}
                        }
                    }
                }
            }

            // Recurse to syntax children
            let children = cpg.ordered_syntax_children(node);
            for child in children {
                collect_analysis_edges_recursive(cpg, child, edges_to_remove);
            }
        }

        collect_analysis_edges_recursive(self, root, &mut to_remove);

        debug!(
            "[CLEAN ANALYSIS EDGES] Removing {} analysis edges from subtree rooted at {:?}",
            to_remove.len(),
            root
        );
        // Remove collected edges
        for edge_id in to_remove {
            self.remove_edge(edge_id);
        }
    }

    /// Replace subtree completely
    fn rebuild_subtree(&mut self, cpg_node: NodeId, cst_node: &tree_sitter::Node) -> NodeId {
        trace!("[REBUILD] Rebuilding subtree for node {:?}", cpg_node);

        // 1. Preserve parent connections by finding the parent edge and check if this is the root
        let parent_edge = self
            .get_incoming_edges(cpg_node)
            .iter()
            .find(|e| e.type_ == EdgeType::SyntaxChild)
            .map(|e| e.from);

        let is_root_node = self.root.map_or(false, |root| root == cpg_node);

        // 2. Remove the old subtree completely
        self.remove_subtree(cpg_node).ok();

        // 3. Reconstruct using the pre/post translate logic
        let (new_node_id, node_type) = match pre_translate_node(self, cst_node) {
            Ok((id, typ)) => (id, typ),
            Err(e) => {
                warn!("[REBUILD] Failed to pre-translate node: {}", e);
                return cpg_node;
            }
        };

        // 4. Recursively build children using the proper translate approach
        let mut cst_cursor = cst_node.walk();
        if cst_cursor.goto_first_child() {
            let mut children = Vec::new();
            loop {
                let child_cst_node = cst_cursor.node();

                // Use pre_translate for each child
                let (child_id, child_type) = match pre_translate_node(self, &child_cst_node) {
                    Ok((id, typ)) => (id, typ),
                    Err(e) => {
                        warn!("[REBUILD] Failed to pre-translate child node: {}", e);
                        if !cst_cursor.goto_next_sibling() {
                            break;
                        }
                        continue;
                    }
                };

                self.add_edge(crate::cpg::Edge {
                    from: new_node_id,
                    to: child_id,
                    type_: EdgeType::SyntaxChild,
                    properties: std::collections::HashMap::new(),
                });

                children.push(child_id);

                // Recursively handle grandchildren if they exist
                if child_cst_node.child_count() > 0 {
                    // Update child_id to the new ID returned by recursive rebuild
                    let new_child_id = self.rebuild_subtree(child_id, &child_cst_node);
                    // Update our children list and edge if needed
                    if new_child_id != child_id {
                        // Find and remove the old edge
                        let edge_ids_to_remove: Vec<_> = self
                            .outgoing
                            .get(&new_node_id)
                            .map(|edge_ids| {
                                edge_ids
                                    .iter()
                                    .filter(|&&edge_id| {
                                        if let Some(edge) = self.edges.get(edge_id) {
                                            edge.to == child_id
                                                && edge.type_ == EdgeType::SyntaxChild
                                        } else {
                                            false
                                        }
                                    })
                                    .copied()
                                    .collect()
                            })
                            .unwrap_or_default();

                        for edge_id in edge_ids_to_remove {
                            self.remove_edge(edge_id);
                        }

                        self.add_edge(crate::cpg::Edge {
                            from: new_node_id,
                            to: new_child_id,
                            type_: EdgeType::SyntaxChild,
                            properties: std::collections::HashMap::new(),
                        });

                        // Update the child in our children list
                        if let Some(last_idx) = children.len().checked_sub(1) {
                            children[last_idx] = new_child_id;
                        }
                    }
                }

                // Apply post-translation logic for the child
                post_translate_node(self, child_type, child_id, &child_cst_node);

                if !cst_cursor.goto_next_sibling() {
                    break;
                }
            }

            // Create sibling edges for children
            for i in 0..(children.len().saturating_sub(1)) {
                self.add_edge(crate::cpg::Edge {
                    from: children[i],
                    to: children[i + 1],
                    type_: EdgeType::SyntaxSibling,
                    properties: std::collections::HashMap::new(),
                });
            }
        }

        // 5. Apply post-translation logic for the root node
        post_translate_node(self, node_type, new_node_id, cst_node);

        // 6. Reattach to parent if it exists, or set as root only if it was originally the root
        if let Some(parent) = parent_edge {
            self.add_edge(crate::cpg::Edge {
                from: parent,
                to: new_node_id,
                type_: EdgeType::SyntaxChild,
                properties: std::collections::HashMap::new(),
            });
        } else if is_root_node {
            // Only set as root if this node was originally the root
            self.set_root(new_node_id);
        }
        // If no parent and not originally root, this node will be orphaned,
        // which is expected for some incremental update scenarios

        new_node_id
    }

    /// Calculate sequence alignment operations (simplified LCS-based approach)
    fn calculate_sequence_alignment(
        &self,
        cpg_children: &[NodeId],
        cst_children: &[tree_sitter::Node],
    ) -> Vec<EditOperation> {
        let mut operations = Vec::new();

        // Simple approach: match by kind and position, generate operations
        let cpg_len = cpg_children.len();
        let cst_len = cst_children.len();

        let mut cpg_idx = 0;
        let mut cst_idx = 0;

        while cpg_idx < cpg_len || cst_idx < cst_len {
            if cpg_idx >= cpg_len {
                // Only CST children left - insert them
                operations.push(EditOperation::Insert { cst_index: cst_idx });
                cst_idx += 1;
            } else if cst_idx >= cst_len {
                // Only CPG children left - delete them
                operations.push(EditOperation::Delete { cpg_index: cpg_idx });
                cpg_idx += 1;
            } else {
                // Both exist - check if they match
                let cpg_node = self.get_node_by_id(&cpg_children[cpg_idx]).unwrap();
                let cpg_kind = cpg_node
                    .properties
                    .get("raw_kind")
                    .cloned()
                    .unwrap_or_else(|| cpg_node.type_.to_string());
                let cst_kind = cst_children[cst_idx].kind().to_string();

                if cpg_kind == cst_kind {
                    // Match - modify if needed
                    operations.push(EditOperation::Modify {
                        cpg_index: cpg_idx,
                        cst_index: cst_idx,
                    });
                    cpg_idx += 1;
                    cst_idx += 1;
                } else {
                    // Look ahead to see if we can find a match
                    let mut found_match = false;

                    // Look for the CST kind in remaining CPG children
                    for look_cpg in (cpg_idx + 1)..cpg_len {
                        let look_node = self.get_node_by_id(&cpg_children[look_cpg]).unwrap();
                        let look_kind = look_node
                            .properties
                            .get("raw_kind")
                            .cloned()
                            .unwrap_or_else(|| look_node.type_.to_string());
                        if look_kind == cst_kind {
                            // Found match ahead - delete current CPG child
                            operations.push(EditOperation::Delete { cpg_index: cpg_idx });
                            cpg_idx += 1;
                            found_match = true;
                            break;
                        }
                    }

                    if !found_match {
                        // No match found - insert CST child
                        operations.push(EditOperation::Insert { cst_index: cst_idx });
                        cst_idx += 1;
                    }
                }
            }
        }

        operations
    }

    /// Apply edit operations to transform CPG children to match CST children
    fn apply_edit_operations(
        &mut self,
        parent: NodeId,
        cpg_children: &[NodeId],
        cst_children: &[tree_sitter::Node],
        operations: Vec<EditOperation>,
    ) {
        // Apply operations
        for operation in operations {
            match operation {
                EditOperation::Insert { cst_index } => {
                    let cst_child = &cst_children[cst_index];

                    // Use pre_translate to create the node properly
                    let (id, node_type) = match pre_translate_node(self, cst_child) {
                        Ok((id, typ)) => (id, typ),
                        Err(e) => {
                            warn!(
                                "[SURGICAL UPDATE] Failed to pre-translate inserted node: {}",
                                e
                            );
                            continue;
                        }
                    };

                    self.add_edge(crate::cpg::Edge {
                        from: parent,
                        to: id,
                        type_: EdgeType::SyntaxChild,
                        properties: std::collections::HashMap::new(),
                    });

                    // Apply post-translation logic
                    post_translate_node(self, node_type, id, cst_child);
                }
                EditOperation::Delete { cpg_index } => {
                    if cpg_index < cpg_children.len() {
                        self.remove_subtree(cpg_children[cpg_index]).ok();
                    }
                }
                EditOperation::Modify {
                    cpg_index,
                    cst_index,
                } => {
                    if cpg_index < cpg_children.len() && cst_index < cst_children.len() {
                        let cpg_id = cpg_children[cpg_index];
                        let cst_child = &cst_children[cst_index];

                        // Update the node properties, span, and apply post-translation logic
                        let new_type = self.get_language().map_node_kind(cst_child.kind());
                        if let Some(node) = self.get_node_by_id_mut(&cpg_id) {
                            node.properties
                                .insert("raw_kind".to_string(), cst_child.kind().to_string());
                            node.type_ = new_type.clone();
                        }
                        self.spatial_index.edit(
                            cpg_id,
                            cst_child.start_byte(),
                            cst_child.end_byte(),
                        );

                        // Apply post-translation logic for updated node
                        post_translate_node(self, new_type, cpg_id, cst_child);
                    }
                }
            }
        }

        // After applying operations, ensure SyntaxSibling edges reflect the
        // current source ordering of SyntaxChild children. Surgical inserts
        // and deletes may leave sibling edges out-of-date which breaks
        // ordering-dependent logic (and can cause missing nodes in diffs).
        // Strategy: gather all current SyntaxChild children of `parent`,
        // sort them by their start byte from the spatial index, remove any
        // existing SyntaxSibling edges among them, and re-add sibling edges
        // to form a proper chain.
        let mut children: Vec<NodeId> = self
            .get_deterministic_sorted_outgoing_edges(parent)
            .into_iter()
            .filter(|e| e.type_ == EdgeType::SyntaxChild)
            .map(|e| e.to)
            .collect();

        if children.len() > 1 {
            // Sort by start byte (fallback to node id order if no span)
            children.sort_by_key(|&child| {
                self.spatial_index
                    .get_node_span(child)
                    .map(|(s, _)| s)
                    .unwrap_or(usize::MAX)
            });

            // Build set for quick membership tests
            let children_set: std::collections::HashSet<_> = children.iter().copied().collect();

            // Remove existing SyntaxSibling edges among these children
            // Collect edge ids to remove first to avoid borrow conflicts
            let mut sibling_edges_to_remove = Vec::new();
            for &child in &children {
                for edge in self.get_deterministic_sorted_outgoing_edges(child) {
                    if edge.type_ == EdgeType::SyntaxSibling && children_set.contains(&edge.to) {
                        // Find the EdgeId for this edge
                        if let Some((edge_id, _)) = self.edges.iter().find(|(_, e)| {
                            e.from == child && e.to == edge.to && e.type_ == EdgeType::SyntaxSibling
                        }) {
                            sibling_edges_to_remove.push(edge_id);
                        }
                    }
                }
            }

            for edge_id in sibling_edges_to_remove {
                self.remove_edge(edge_id);
            }

            // Recreate sibling edges in order
            for i in 0..(children.len().saturating_sub(1)) {
                self.add_edge(crate::cpg::Edge {
                    from: children[i],
                    to: children[i + 1],
                    type_: EdgeType::SyntaxSibling,
                    properties: std::collections::HashMap::new(),
                });
            }
        }
    }

    /// Incrementally update the CPG from the CST edits
    pub fn incremental_update(
        &mut self,
        edits: Vec<SourceEdit>,
        changed_ranges: impl ExactSizeIterator<Item = tree_sitter::Range>,
        new_tree: &tree_sitter::Tree,
        new_source: Vec<u8>,
    ) {
        let changed_ranges = changed_ranges.collect::<Vec<_>>();

        debug!(
            "[INCREMENTAL UPDATE] [PARSE EDITS] Update with {} edits and {} changed ranges",
            edits.len(),
            changed_ranges.len()
        );

        self.set_source(new_source);

        let source_len = self.get_source().len();
        let root_node = new_tree.root_node();
        if root_node.start_byte() > source_len || root_node.end_byte() > source_len {
            warn!(
                "[INCREMENTAL UPDATE] [PARSE EDITS] Invalid root node range ({}, {}) for source length {}.",
                root_node.start_byte(),
                root_node.end_byte(),
                source_len
            );
        }

        let mut dirty_nodes = HashMap::new();
        debug!(
            "[INCREMENTAL UPDATE] [PARSE EDITS] TS Changed ranges: {:?}",
            changed_ranges.clone()
        );
        for range in changed_ranges {
            if let Some(node_id) =
                self.get_smallest_node_id_containing_range(range.start_byte, range.end_byte)
            {
                trace!(
                    "[INCREMENTAL UPDATE] [PARSE EDITS] Found node {:?} for range {:?}",
                    node_id, range
                );
                if let Some(node) = self.get_node_by_id(&node_id) {
                    trace!(
                        "[INCREMENTAL UPDATE] [PARSE EDITS] Node details: {:?}",
                        node
                    );
                }
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
                    "[INCREMENTAL UPDATE] [PARSE EDITS] No node found for changed range: {:?}",
                    (range.start_byte, range.end_byte)
                );

                // When a changed range doesn't map to an existing node, this usually means
                // new structure was added. Find the closest parent node that can contain this range.
                if let Some(parent_node_id) =
                    self.find_parent_node_for_range(range.start_byte, range.end_byte)
                {
                    debug!(
                        "[INCREMENTAL UPDATE] [PARSE EDITS] Found parent node {:?} for orphaned range {:?}",
                        parent_node_id, range
                    );
                    dirty_nodes.insert(
                        parent_node_id,
                        (
                            range.start_byte,
                            range.end_byte,
                            range.start_byte,
                            range.end_byte,
                        ),
                    );
                } else {
                    warn!(
                        "[INCREMENTAL UPDATE] [PARSE EDITS] No parent node found for orphaned range: {:?}",
                        (range.start_byte, range.end_byte)
                    );
                }
            }
        }

        debug!(
            "[INCREMENTAL UPDATE] [PARSE EDITS] Textual edits: {:?}",
            edits
        );
        for edit in edits {
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
                    "[INCREMENTAL UPDATE] [PARSE EDITS] No node found for edit range: {:?}",
                    (edit.old_start, edit.old_end)
                );
            }
        }

        debug!(
            "[INCREMENTAL UPDATE] [FILTERING] Filtering {} dirty nodes",
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
            "[INCREMENTAL UPDATE] [FILTERING] Merged {} overlapping ranges into {} ranges",
            dirty_nodes.len(),
            merged_ranges.len()
        );

        // Track orphaned parent nodes separately to ensure they get processed
        let mut orphaned_parent_nodes = std::collections::HashSet::new();

        // For each merged range, find the appropriate node that contains it
        let containing_nodes_with_ranges: Vec<(NodeId, (usize, usize, usize, usize))> = merged_ranges
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
                        // This must match the priority system in spatial_index.rs
                        let priority_weight = match &node.type_ {
                            // Highest priority: Structural containers (translation units)
                            NodeType::TranslationUnit => 0,

                            // High priority: Control flow structures that can change significantly
                            NodeType::Branch { .. } => 1,
                            NodeType::Loop { .. } => 1,

                            // Medium priority: Functions and major structural elements
                            NodeType::Function { .. } => 2,
                            NodeType::Statement => 20,
                            NodeType::Block => 30,

                            // Lower priority: Expression-level constructs
                            NodeType::Expression => 100,
                            NodeType::Call => 100,
                            NodeType::Return => 100,

                            // Lowest priority: Leaf nodes and language-specific constructs
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
                    trace!(
                        "[INCREMENTAL UPDATE] [FILTERING] Found containing node {:?} for merged range {:?}",
                        node_id, range
                    );
                    Some((node_id, range))
                } else {
                    warn!(
                        "[INCREMENTAL UPDATE] [FILTERING] No containing node found for merged range {:?}",
                        range
                    );
                    // For orphaned ranges, try to find a parent node that can be rebuilt
                    if let Some(parent_node_id) = self.find_parent_node_for_range(range.0, range.1)
                    {
                        warn!(
                            "[INCREMENTAL UPDATE] [FILTERING] Found orphaned parent node {:?} for range {:?}",
                            parent_node_id, range
                        );
                        orphaned_parent_nodes.insert(parent_node_id);

                        // Expand the range to cover the parent node's span
                        if let Some((parent_start, parent_end)) =
                            self.spatial_index.get_node_span(parent_node_id)
                        {
                            let expanded_range = (
                                std::cmp::min(range.0, parent_start),
                                std::cmp::max(range.1, parent_end),
                                std::cmp::min(range.2, parent_start),
                                std::cmp::max(range.3, parent_end),
                            );
                            Some((parent_node_id, expanded_range))
                        } else {
                            Some((parent_node_id, range))
                        }
                    } else {
                        None
                    }
                }
            })
            .collect();

        // First attempt: try to surgically update nodes that can be updated in-place
        // If a node can't be surgically updated, collect it as a rebuild candidate and
        // proceed to merge/expand those ranges as before.
        let thresholds = UpdateThresholds::default();

        let mut rebuild_candidates: Vec<(NodeId, (usize, usize, usize, usize))> = Vec::new();
        // Surgical candidates will be applied after we compute merged rebuild ranges
        // to avoid mutating spatial_index before pairing merged ranges to containers.
        let mut early_surgical_candidates: Vec<(NodeId, tree_sitter::Node)> = Vec::new();
        let mut early_surgical_ids: HashSet<NodeId> = HashSet::new();
        // Track structures/functions updated by surgical updates so we don't
        // double-work them later.
        let mut updated_structures: Vec<NodeId> = Vec::new();
        let mut updated_functions: Vec<NodeId> = Vec::new();

        for (node_id, range) in &containing_nodes_with_ranges {
            // If this node was identified as an orphaned parent for a changed range,
            // prefer a full rebuild — orphaned parents usually indicate newly added
            // structure that surgical updates may not handle correctly.
            if orphaned_parent_nodes.contains(node_id) {
                rebuild_candidates.push((*node_id, *range));
                continue;
            }
            // Pair to CST node in the new tree (special-case root)
            let cst_node_opt = if self.root.map_or(false, |root| root == *node_id) {
                Some(new_tree.root_node())
            } else {
                new_tree
                    .root_node()
                    .descendant_for_byte_range(range.0, range.1)
            };

            if let Some(cst_node) = cst_node_opt {
                let (should_rebuild, metrics) =
                    self.should_rebuild(*node_id, &cst_node, &thresholds);
                trace!(
                    "[INCREMENTAL UPDATE] [EARLY DECIDE] Node {:?}: should_rebuild={}, metrics={:?}",
                    node_id, should_rebuild, metrics
                );

                if !should_rebuild {
                    // Defer actual surgical update until after we compute merged rebuild ranges
                    trace!(
                        "[INCREMENTAL UPDATE] [EARLY SURGICAL] Queuing node {:?} for surgical update",
                        node_id
                    );
                    early_surgical_candidates.push((*node_id, cst_node));
                    early_surgical_ids.insert(*node_id);
                } else {
                    // Needs rebuild, collect for merge/expand
                    rebuild_candidates.push((*node_id, *range));
                }
            } else {
                // No CST pairing - schedule for rebuild
                rebuild_candidates.push((*node_id, *range));
            }
        }

        // Now continue with the existing merge/expand logic but only for rebuild candidates
        // Group overlapping ranges from rebuild_candidates
        let mut merged_ranges: Vec<(usize, usize, usize, usize)> = Vec::new();

        for (_, range) in rebuild_candidates.iter() {
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
            "[INCREMENTAL UPDATE] [FILTERING] From {} rebuild candidates merged into {} ranges",
            rebuild_candidates.len(),
            merged_ranges.len()
        );

        // Apply queued surgical updates now that rebuild ranges are computed.
        // Doing this earlier could mutate spans used for pairing merged ranges.
        for (node_id, cst_node) in early_surgical_candidates.drain(..) {
            trace!(
                "[INCREMENTAL UPDATE] [EARLY SURGICAL APPLY] Applying surgical update for {:?}",
                node_id
            );
            // Clean analysis edges before surgical update to avoid stale PDData/CF edges
            // interfering with post-translation and comparison.
            self.clean_analysis_edges(node_id);
            self.update_in_place_pairwise(node_id, &cst_node);

            let structure = crate::languages::get_container_parent(self, node_id);
            if !updated_structures.contains(&structure) {
                updated_structures.push(structure);
            }
            if let Some(function) = crate::languages::get_containing_function(self, node_id) {
                if !updated_functions.contains(&function) {
                    updated_functions.push(function);
                }
            }
        }

        // Add any orphaned parent nodes that weren't already included
        for node_id in orphaned_parent_nodes.iter().cloned() {
            if let Some((start, end)) = self.spatial_index.get_node_span(node_id) {
                if !merged_ranges.iter().any(|mr| mr.0 <= start && end <= mr.1) {
                    merged_ranges.push((start, end, start, end));
                    trace!(
                        "[INCREMENTAL UPDATE] [FILTERING] Added orphaned parent node {:?} to rebuild ranges",
                        node_id
                    );
                }
            }
        }

        // Pair merged ranges back to containing nodes to produce the rebuild update plan
        let mut containing_nodes_with_ranges: Vec<(NodeId, (usize, usize, usize, usize))> =
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

                    // Choose the most appropriate node to rehydrate (same logic as above)
                    let containing_node = candidates
                        .into_iter()
                        .filter_map(|node_id| {
                            // Skip nodes already queued for surgical updates
                            if early_surgical_ids.contains(&node_id) {
                                return None;
                            }
                            let node = self.get_node_by_id(&node_id)?;
                            let node_range = self.spatial_index.get_node_span(node_id)?;
                            let range_size = node_range.1 - node_range.0;

                            // Assign priority weights - lower values = higher priority
                            let priority_weight = match &node.type_ {
                                NodeType::TranslationUnit => 0,
                                NodeType::Branch { .. } => 1,
                                NodeType::Loop { .. } => 1,
                                NodeType::Function { .. } => 2,
                                NodeType::Statement => 20,
                                NodeType::Block => 30,
                                NodeType::Expression => 100,
                                NodeType::Call => 100,
                                NodeType::Return => 100,
                                NodeType::Identifier { .. } => 1000,
                                NodeType::Comment => 1000,
                                NodeType::Type => 500,
                                NodeType::LanguageImplementation(_) => 800,
                                _ => 200,
                            };

                            Some((node_id, priority_weight + range_size))
                        })
                        .min_by_key(|(_, weight)| *weight)
                        .map(|(node_id, _)| node_id);

                    if let Some(node_id) = containing_node {
                        trace!(
                            "[INCREMENTAL UPDATE] [FILTERING] Found containing node {:?} for merged range {:?}",
                            node_id, range
                        );
                        Some((node_id, range))
                    } else {
                        warn!(
                            "[INCREMENTAL UPDATE] [FILTERING] No containing node found for merged range {:?}",
                            range
                        );
                        None
                    }
                })
                .collect();

        // Ensure any orphaned parent nodes are included in the rebuild plan if they
        // weren't matched by spatial_index pairing above. This covers cases where
        // ranges moved or the index lookup missed the intended parent.
        for node_id in orphaned_parent_nodes.clone().into_iter() {
            if !containing_nodes_with_ranges
                .iter()
                .any(|(id, _)| *id == node_id)
            {
                if let Some((start, end)) = self.spatial_index.get_node_span(node_id) {
                    containing_nodes_with_ranges.push((node_id, (start, end, start, end)));
                    trace!(
                        "[INCREMENTAL UPDATE] [FILTERING] Added orphaned parent node {:?} to containing_nodes_with_ranges",
                        node_id
                    );
                }
            }
        }

        debug!(
            "[INCREMENTAL UPDATE] [FILTERING] After deduplication: {} containing nodes to consider for rebuild",
            containing_nodes_with_ranges.len()
        );

        // To avoid borrow checker issues, collect update info first
        let mut update_plan = Vec::new();
        for (id, _pos) in &containing_nodes_with_ranges {
            let node_span = self.spatial_index.get_node_span(*id);
            if let Some((start, end)) = node_span {
                // Special case: if this is the root node, use the root of the new tree
                let cst_node = if self.root.map_or(false, |root| root == *id) {
                    trace!(
                        "[INCREMENTAL UPDATE] [CST PAIRING] Using new tree root for CPG root node {:?}",
                        id
                    );

                    let root_node = new_tree.root_node();
                    // Validate that the tree root's ranges are within the new source bounds
                    let source_len = self.get_source().len();
                    if root_node.start_byte() > source_len || root_node.end_byte() > source_len {
                        warn!(
                            "[INCREMENTAL UPDATE] [CST PAIRING] Tree root node range ({}, {}) exceeds source length {}.",
                            root_node.start_byte(),
                            root_node.end_byte(),
                            source_len
                        );
                    }

                    root_node
                } else if let Some(cst_node) =
                    new_tree.root_node().descendant_for_byte_range(start, end)
                {
                    trace!(
                        "[INCREMENTAL UPDATE] [CST PAIRING] Found CST node for CPG node {:?} with span ({}, {}): kind={}, cst_span=({}, {})",
                        id,
                        start,
                        end,
                        cst_node.kind(),
                        cst_node.start_byte(),
                        cst_node.end_byte()
                    );
                    cst_node
                } else {
                    warn!(
                        "[INCREMENTAL UPDATE] [CST PAIRING] No CST node found for CPG node {:?} in new tree",
                        id
                    );
                    continue;
                };

                update_plan.push((*id, cst_node));
            } else {
                warn!(
                    "[INCREMENTAL UPDATE] [CST PAIRING] No span found for CPG node {:?}",
                    id
                );
            }
        }
        for (id, cst_node) in &update_plan {
            let (should_rebuild, metrics) = self.should_rebuild(*id, cst_node, &thresholds);
            trace!(
                "[INCREMENTAL UPDATE] [REBUILD] [DECIDE] Node {:?}: should_rebuild={}, metrics={:?}",
                id, should_rebuild, metrics
            );

            let new_id;
            if should_rebuild {
                trace!(
                    "[INCREMENTAL UPDATE] [REBUILD] [FULL] Rebuilding subtree for node {:?} ({}% children changed, {} ops, depth {})",
                    id,
                    metrics.children_change_percentage * 100.0,
                    metrics.operations_count,
                    metrics.subtree_depth
                );
                self.clean_analysis_edges(*id);
                new_id = self.rebuild_subtree(*id, cst_node);
            } else {
                trace!(
                    "[INCREMENTAL UPDATE] [REBUILD] [SURGICAL] Updating node {:?} ({}% children changed, {} ops, depth {})",
                    id,
                    metrics.children_change_percentage * 100.0,
                    metrics.operations_count,
                    metrics.subtree_depth
                );
                // Clean analysis edges before performing surgical updates so we can
                // recompute them deterministically afterwards.
                self.clean_analysis_edges(*id);
                self.update_in_place_pairwise(*id, cst_node);
                new_id = *id;
            }

            let structure = crate::languages::get_container_parent(self, new_id);
            if !updated_structures.contains(&structure) {
                updated_structures.push(structure);
            }
            if let Some(function) = crate::languages::get_containing_function(self, new_id) {
                if !updated_functions.contains(&function) {
                    updated_functions.push(function);
                }
            }
        }

        debug!(
            "[INCREMENTAL UPDATE] [ANALYSIS EDGES] Recomputing control flow for updated structures: {:?}",
            updated_structures
        );
        for structure in &updated_structures {
            trace!(
                "[INCREMENTAL UPDATE] [ANALYSIS EDGES] Running cf_pass on {:?}",
                structure
            );
            if let Err(e) = cf_pass(self, *structure) {
                warn!(
                    "[INCREMENTAL UPDATE] [ANALYSIS EDGES] Failed to recompute control flow for node {:?}: {}",
                    structure, e
                );
            }
        }
        debug!(
            "[INCREMENTAL UPDATE] [ANALYSIS EDGES] Recomputing data dependence for updated functions: {:?}",
            updated_functions
        );
        for function in &updated_functions {
            trace!(
                "[INCREMENTAL UPDATE] [ANALYSIS EDGES] Running data_dep_pass on {:?}",
                function
            );
            if let Err(e) = data_dep_pass(self, *function) {
                warn!(
                    "[INCREMENTAL UPDATE] [ANALYSIS EDGES] Failed to recompute data dependence for node {:?}: {}",
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

        // Get current node state
        let cpg_node_ref = self.get_node_by_id(&cpg_node);
        if cpg_node_ref.is_none() {
            return; // Node no longer exists
        }

        let current_type = cpg_node_ref.unwrap().type_.clone();
        let current_properties = cpg_node_ref.unwrap().properties.clone();
        let current_span = self.spatial_index.get_node_span(cpg_node);

        let cst_kind = cst_node.kind();
        let new_type = lang.map_node_kind(cst_kind);
        let cst_start = cst_node.start_byte();
        let cst_end = cst_node.end_byte();

        // Only update if type actually changed
        if current_type != new_type {
            if let Some(node) = self.get_node_by_id_mut(&cpg_node) {
                node.type_ = new_type;
            }
        }

        // Only update span if it actually changed
        if current_span != Some((cst_start, cst_end)) {
            self.spatial_index.edit(cpg_node, cst_start, cst_end);
        }

        // Only update raw_kind if it actually changed
        let new_raw_kind = cst_kind.to_string();
        let current_raw_kind = current_properties.get("raw_kind");
        if current_raw_kind != Some(&new_raw_kind) {
            if let Some(node) = self.get_node_by_id_mut(&cpg_node) {
                node.properties.insert("raw_kind".to_string(), new_raw_kind);

                // Preserve semantic properties that language analysis added
                for (key, value) in current_properties {
                    if key != "raw_kind" && !node.properties.contains_key(&key) {
                        node.properties.insert(key, value);
                    }
                }
            }
        }

        // Re-run post-translation analysis to update semantic properties (like function names)
        // This is crucial for nodes like functions whose properties depend on their children
        let updated_type = self.get_node_by_id(&cpg_node).unwrap().type_.clone();
        crate::languages::post_translate_node(self, updated_type, cpg_node, cst_node);

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

        let operations = self.calculate_sequence_alignment(&cpg_children, &cst_children);
        debug!(
            "[SURGICAL UPDATE] Calculated {} operations for node {:?}",
            operations.len(),
            cpg_node
        );

        self.apply_edit_operations(cpg_node, &cpg_children, &cst_children, operations);

        let final_children = self.ordered_syntax_children(cpg_node);
        let mut cst_cursor = cst_node.walk();
        if cst_cursor.goto_first_child() {
            let mut cst_child_index = 0;
            for &child_id in &final_children {
                if cst_child_index < cst_children.len() {
                    let cst_child = &cst_children[cst_child_index];
                    // Only recurse if this child still exists and matches
                    let child_cpg_node = self.get_node_by_id(&child_id);
                    if let Some(cpg_child) = child_cpg_node {
                        let cpg_kind = cpg_child
                            .properties
                            .get("raw_kind")
                            .cloned()
                            .unwrap_or_else(|| cpg_child.type_.to_string());
                        let cst_kind = cst_child.kind().to_string();

                        if cpg_kind == cst_kind {
                            debug!(
                                "[SURGICAL UPDATE] Recursing into child {:?} of type {}",
                                child_id, cpg_kind
                            );
                            self.update_in_place_pairwise(child_id, cst_child);

                            // Re-run post-translation for the child node to update its semantic properties
                            let child_type = self.get_node_by_id(&child_id).unwrap().type_.clone();
                            debug!(
                                "[SURGICAL UPDATE] Calling post-translation for child {:?}",
                                child_id
                            );
                            crate::languages::post_translate_node(
                                self, child_type, child_id, cst_child,
                            );
                        } else {
                            debug!(
                                "[SURGICAL UPDATE] Skipping child {:?} - kind mismatch: cpg={}, cst={}",
                                child_id, cpg_kind, cst_kind
                            );
                        }
                    }
                    cst_child_index += 1;
                }
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

        // Remove the node and its spatial index entry first
        self.nodes.remove(root);
        self.spatial_index.delete(root);

        // Collect edge ids directly from adjacency sets instead of scanning the whole edges map.
        // This avoids iterating self.edges and the expensive ::next calls on large maps.
        let mut edges_to_remove: HashSet<super::EdgeId> = HashSet::new();

        if let Some(out_set) = self.outgoing.remove(&root) {
            edges_to_remove.extend(out_set.into_iter());
        }
        if let Some(in_set) = self.incoming.remove(&root) {
            edges_to_remove.extend(in_set.into_iter());
        }

        // Remove each edge and update adjacency lists of neighbor nodes
        for edge_id in edges_to_remove {
            if let Some(edge) = self.edges.remove(edge_id) {
                if edge.from != root {
                    if let Some(out_edges) = self.outgoing.get_mut(&edge.from) {
                        out_edges.remove(&edge_id);
                    }
                }
                if edge.to != root {
                    if let Some(in_edges) = self.incoming.get_mut(&edge.to) {
                        in_edges.remove(&edge_id);
                    }
                }
            }
        }

        Ok(())
    }
}
