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
            max_children_change_percentage: 0.3, // Commit-level batch processing
            max_surgical_depth: 6,               // Reduced depth limit for batches
            max_operations_count: 8,             // Reduced operation limit for commit optimization
        }
    }
}
/// Metrics for deciding between surgical update vs full rebuild
#[derive(Debug)]
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

        // Remove collected edges
        for edge_id in to_remove {
            self.remove_edge(edge_id);
        }
    }

    /// Phase 3B: Full Rebuild - Replace subtree completely
    fn rebuild_subtree(&mut self, cpg_node: NodeId, cst_node: &tree_sitter::Node) {
        debug!("[REBUILD] Rebuilding subtree for node {:?}", cpg_node);

        // 1. Preserve parent connections by finding the parent edge
        let parent_edge = self
            .get_incoming_edges(cpg_node)
            .iter()
            .find(|e| e.type_ == EdgeType::SyntaxChild)
            .map(|e| e.from);

        // 2. Remove the old subtree completely
        self.remove_subtree(cpg_node).ok();

        // 3. Reconstruct using the existing language mapping logic
        let lang = self.get_language().clone();
        let new_type = lang.map_node_kind(cst_node.kind());
        let mut new_node = crate::cpg::Node {
            type_: new_type,
            properties: std::collections::HashMap::new(),
        };
        new_node
            .properties
            .insert("raw_kind".to_string(), cst_node.kind().to_string());

        let new_node_id = self.add_node(new_node, cst_node.start_byte(), cst_node.end_byte());

        // 4. Recursively build children using simple approach
        let mut cst_cursor = cst_node.walk();
        if cst_cursor.goto_first_child() {
            let mut children = Vec::new();
            loop {
                let child_node = cst_cursor.node();
                let child_type = lang.map_node_kind(child_node.kind());
                let mut child = crate::cpg::Node {
                    type_: child_type,
                    properties: std::collections::HashMap::new(),
                };
                child
                    .properties
                    .insert("raw_kind".to_string(), child_node.kind().to_string());

                let child_id = self.add_node(child, child_node.start_byte(), child_node.end_byte());

                self.add_edge(crate::cpg::Edge {
                    from: new_node_id,
                    to: child_id,
                    type_: EdgeType::SyntaxChild,
                    properties: std::collections::HashMap::new(),
                });

                children.push(child_id);

                if child_node.child_count() > 0 {
                    self.build_children_from_cst(child_id, &child_node);
                }

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

        // 5. Reattach to parent if it exists
        if let Some(parent) = parent_edge {
            self.add_edge(crate::cpg::Edge {
                from: parent,
                to: new_node_id,
                type_: EdgeType::SyntaxChild,
                properties: std::collections::HashMap::new(),
            });
        } else {
            // This was the root node
            self.set_root(new_node_id);
        }
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
        let lang = self.get_language().clone();

        // Apply operations
        for operation in operations {
            match operation {
                EditOperation::Insert { cst_index } => {
                    let cst_child = &cst_children[cst_index];
                    let type_ = lang.map_node_kind(cst_child.kind());
                    let mut node = crate::cpg::Node {
                        type_,
                        properties: std::collections::HashMap::new(),
                    };
                    node.properties
                        .insert("raw_kind".to_string(), cst_child.kind().to_string());

                    let id = self.add_node(node, cst_child.start_byte(), cst_child.end_byte());
                    self.add_edge(crate::cpg::Edge {
                        from: parent,
                        to: id,
                        type_: EdgeType::SyntaxChild,
                        properties: std::collections::HashMap::new(),
                    });
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

                        // Update the node properties and span
                        if let Some(node) = self.get_node_by_id_mut(&cpg_id) {
                            node.properties
                                .insert("raw_kind".to_string(), cst_child.kind().to_string());
                            let new_type = lang.map_node_kind(cst_child.kind());
                            node.type_ = new_type;
                        }
                        self.spatial_index.edit(
                            cpg_id,
                            cst_child.start_byte(),
                            cst_child.end_byte(),
                        );
                    }
                }
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

        // Phase 1: Clean Slate - Remove all analysis edges from dirty subtrees
        debug!(
            "[INCREMENTAL UPDATE] Phase 1: Cleaning analysis edges from {} dirty nodes",
            update_plan.len()
        );
        for (id, _cst_node) in &update_plan {
            self.clean_analysis_edges(*id);
        }

        // Phase 2: Decision Framework - Decide surgical vs rebuild for each node
        let thresholds = UpdateThresholds::default();
        for (id, cst_node) in &update_plan {
            let (should_rebuild, metrics) = self.should_rebuild(*id, cst_node, &thresholds);
            debug!(
                "[INCREMENTAL UPDATE] Node {:?}: should_rebuild={}, metrics={:?}",
                id, should_rebuild, metrics
            );

            if should_rebuild {
                // Phase 3B: Full Rebuild
                debug!(
                    "[INCREMENTAL UPDATE] Rebuilding subtree for node {:?} ({}% children changed, {} ops, depth {})",
                    id,
                    metrics.children_change_percentage * 100.0,
                    metrics.operations_count,
                    metrics.subtree_depth
                );
                self.rebuild_subtree(*id, cst_node);
            } else {
                // Phase 3A: Surgical Update
                debug!(
                    "[INCREMENTAL UPDATE] Surgical update for node {:?} ({}% children changed, {} ops, depth {})",
                    id,
                    metrics.children_change_percentage * 100.0,
                    metrics.operations_count,
                    metrics.subtree_depth
                );
                self.update_in_place_pairwise(*id, cst_node);
            }

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
                            self.update_in_place_pairwise(child_id, cst_child);
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
