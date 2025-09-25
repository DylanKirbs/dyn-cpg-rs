use super::{Cpg, CpgError, NodeId, edge::EdgeType, node::NodeType, serialization::SexpSerializer};
use similar::TextDiff;
use std::collections::{HashMap, HashSet};
use tracing::debug;

// --- Comparison Results --- //

/// Detailed result of a CPG comparison
#[derive(Clone, Debug)]
pub enum DetailedComparisonResult<'a> {
    /// The CPGs are semantically equivalent
    Equivalent,
    /// The CPGs have structural differences
    StructuralMismatch {
        /// Functions present only in the left CPG
        only_in_left: Vec<String>,
        /// Functions present only in the right CPG
        only_in_right: Vec<String>,
        /// Functions that exist in both but have differences
        function_mismatches: Vec<FunctionComparisonResult<'a>>,
    },
}

impl PartialEq for DetailedComparisonResult<'_> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (DetailedComparisonResult::Equivalent, DetailedComparisonResult::Equivalent) => true,
            (
                DetailedComparisonResult::StructuralMismatch {
                    only_in_left: left1,
                    only_in_right: right1,
                    function_mismatches: funcs1,
                    ..
                },
                DetailedComparisonResult::StructuralMismatch {
                    only_in_left: left2,
                    only_in_right: right2,
                    function_mismatches: funcs2,
                    ..
                },
            ) => left1 == left2 && right1 == right2 && funcs1 == funcs2,
            _ => false,
        }
    }
}

impl<'a> std::fmt::Display for DetailedComparisonResult<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DetailedComparisonResult::Equivalent => write!(f, "CPGs are equivalent"),
            DetailedComparisonResult::StructuralMismatch {
                only_in_left,
                only_in_right,
                function_mismatches,
            } => {
                writeln!(f, "CPGs have structural differences:")?;

                if !only_in_left.is_empty() {
                    writeln!(f, "  Functions only in left: {:?}", only_in_left)?;
                }
                if !only_in_right.is_empty() {
                    writeln!(f, "  Functions only in right: {:?}", only_in_right)?;
                }
                for mismatch in function_mismatches {
                    writeln!(f, "  Function mismatch: {}", mismatch)?;
                }
                Ok(())
            }
        }
    }
}

/// Result of comparing a single function between two CPGs
#[derive(Debug, Clone)]
pub enum FunctionComparisonResult<'a> {
    /// The functions are equivalent
    Equivalent,
    /// The functions differ
    Mismatch {
        /// The name of the function
        function_name: String,
        /// References to the DetailedComparisonResult's CPGs for context
        left_cpg: &'a Cpg,
        right_cpg: &'a Cpg,
        mismatches: Vec<(Option<NodeId>, Option<NodeId>, String)>,
        l_root: NodeId,
        r_root: NodeId,
    },
}

impl<'a> PartialEq for FunctionComparisonResult<'a> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (FunctionComparisonResult::Equivalent, FunctionComparisonResult::Equivalent) => true,
            (
                FunctionComparisonResult::Mismatch {
                    function_name: fn1, ..
                },
                FunctionComparisonResult::Mismatch {
                    function_name: fn2, ..
                },
            ) => fn1 == fn2,
            _ => false,
        }
    }
}

impl<'a> std::fmt::Display for FunctionComparisonResult<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FunctionComparisonResult::Equivalent => write!(f, "Function is equivalent"),
            FunctionComparisonResult::Mismatch {
                function_name,
                left_cpg,
                right_cpg,
                mismatches,
                l_root,
                r_root,
            } => {
                let max_diff_size = 1024;

                let mut source = left_cpg.source_diff(right_cpg, mismatches);
                let mut sexp = left_cpg.sexp_diff(right_cpg, *l_root, *r_root);

                if source.len() > max_diff_size {
                    source.truncate(max_diff_size);
                    source.push_str("\n... (truncated)");
                }
                if sexp.len() > max_diff_size {
                    sexp.truncate(max_diff_size);
                    sexp.push_str("\n... (truncated)");
                }

                writeln!(f, "Function '{}' has differences:", function_name)?;
                writeln!(
                    f,
                    "  Source Diff:\n    {}",
                    source.replace("\n", "\n    ").trim()
                )?;
                writeln!(
                    f,
                    "  Sexp Diff:\n    {}",
                    sexp.replace("\n", "\n    ").trim()
                )?;
                Ok(())
            }
        }
    }
}

// --- Helper Functions --- //

fn to_sorted_vec(properties: &HashMap<String, String>) -> Vec<(String, String)> {
    let mut vec: Vec<_> = properties
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    vec.sort_by(|a, b| a.0.cmp(&b.0));
    vec
}

// --- CPG Comparison Implementation --- //

pub type RootMismatches = (Option<NodeId>, Option<NodeId>, String);

impl Cpg {
    /// Compare two CPGs for semantic equality
    /// Returns a detailed comparison result indicating structural differences and function-level mismatches
    pub fn compare<'a>(&'a self, other: &'a Cpg) -> Result<DetailedComparisonResult<'a>, CpgError> {
        debug!(
            "[COMPARE] Comparing CPGs: left root = {:?}, right root = {:?}",
            self.get_root(),
            other.get_root()
        );

        let left_root = self.get_root();
        let right_root = other.get_root();

        match (left_root, right_root) {
            (None, None) => Ok(DetailedComparisonResult::Equivalent),
            (None, Some(_)) => Ok(DetailedComparisonResult::StructuralMismatch {
                only_in_left: vec![],
                only_in_right: vec!["root".to_string()],
                function_mismatches: vec![],
            }),
            (Some(_), None) => Ok(DetailedComparisonResult::StructuralMismatch {
                only_in_left: vec!["root".to_string()],
                only_in_right: vec![],
                function_mismatches: vec![],
            }),
            (Some(l_root), Some(r_root)) => {
                let l_node = self.get_node_by_id(&l_root).ok_or_else(|| {
                    CpgError::MissingField(format!(
                        "Node with id {:?} not found in left CPG",
                        l_root
                    ))
                })?;
                let r_node = other.get_node_by_id(&r_root).ok_or_else(|| {
                    CpgError::MissingField(format!(
                        "Node with id {:?} not found in right CPG",
                        r_root
                    ))
                })?;

                // Check if both roots are TranslationUnit nodes
                if l_node.type_ != NodeType::TranslationUnit
                    || r_node.type_ != NodeType::TranslationUnit
                {
                    // If roots aren't TranslationUnits, fall back to subtree comparison
                    let mut visited = std::collections::HashSet::new();
                    let mismatches = self.compare_subtrees(other, l_root, r_root, &mut visited)?;
                    if mismatches.is_empty() {
                        return Ok(DetailedComparisonResult::Equivalent);
                    } else {
                        return {
                            Ok(DetailedComparisonResult::StructuralMismatch {
                                only_in_left: vec![],
                                only_in_right: vec![],
                                function_mismatches: vec![FunctionComparisonResult::Mismatch {
                                    function_name: "root".to_string(),
                                    left_cpg: self,
                                    right_cpg: other,
                                    mismatches,
                                    l_root,
                                    r_root,
                                }],
                            })
                        };
                    }
                }

                // Compare top-level structure
                let left_functions = self.get_top_level_functions(l_root)?;
                let right_functions = other.get_top_level_functions(r_root)?;

                debug!(
                    "[COMPARE] Left functions: {:?} Right functions: {:?}",
                    left_functions.keys().collect::<Vec<_>>(),
                    right_functions.keys().collect::<Vec<_>>()
                );

                let mut only_in_left = Vec::new();
                let mut only_in_right = Vec::new();
                let mut function_mismatches = Vec::new();

                // Find functions only in left CPG
                for name in left_functions.keys() {
                    if !right_functions.contains_key(name) {
                        only_in_left.push(name.clone());
                    }
                }

                // Find functions only in right CPG
                for name in right_functions.keys() {
                    if !left_functions.contains_key(name) {
                        only_in_right.push(name.clone());
                    }
                }

                // Compare functions present in both CPGs
                for (name, left_func_id) in &left_functions {
                    if let Some(right_func_id) = right_functions.get(name) {
                        let mut visited = std::collections::HashSet::new();
                        let mismatches = self.compare_subtrees(
                            other,
                            *left_func_id,
                            *right_func_id,
                            &mut visited,
                        )?;

                        if !mismatches.is_empty() {
                            function_mismatches.push(FunctionComparisonResult::Mismatch {
                                function_name: name.clone(),
                                mismatches,
                                left_cpg: self,
                                right_cpg: other,
                                l_root: *left_func_id,
                                r_root: *right_func_id,
                            });
                        }
                    }
                }

                // Check if there are any differences
                if only_in_left.is_empty()
                    && only_in_right.is_empty()
                    && function_mismatches.is_empty()
                {
                    Ok(DetailedComparisonResult::Equivalent)
                } else {
                    Ok(DetailedComparisonResult::StructuralMismatch {
                        only_in_left,
                        only_in_right,
                        function_mismatches,
                    })
                }
            }
        }
    }

    /// Get all top-level function definitions from a TranslationUnit
    pub fn get_top_level_functions(
        &self,
        root: NodeId,
    ) -> Result<HashMap<String, NodeId>, CpgError> {
        let mut functions = HashMap::new();

        // Get all SyntaxChild edges from the root
        let child_edges = self.get_deterministic_sorted_outgoing_edges(root);

        for edge in child_edges {
            if edge.type_ == EdgeType::SyntaxChild {
                let node = self.get_node_by_id(&edge.to).ok_or_else(|| {
                    CpgError::MissingField(format!("Child node with id {:?} not found", edge.to))
                })?;

                // Check if this child is a Function node
                if let NodeType::Function { .. } = node.type_ {
                    // Try to get the function name from properties
                    let name = node
                        .properties
                        .get("name")
                        .cloned()
                        .unwrap_or_else(|| format!("unnamed_function_{:?}", edge.to));

                    functions.insert(name, edge.to);
                }
            }
        }

        Ok(functions)
    }

    /// Compare two subtrees, updating a list of the NodeIds of the sub-subtrees that are mismatched
    fn compare_subtrees(
        &self,
        other: &Cpg,
        l_root: NodeId,
        r_root: NodeId,
        visited: &mut HashSet<(NodeId, NodeId)>,
    ) -> Result<Vec<RootMismatches>, CpgError> {
        // Avoid re-comparing the same pair of nodes
        if !visited.insert((l_root, r_root)) {
            return Ok(vec![]);
        }

        let l_node = self.get_node_by_id(&l_root).ok_or_else(|| {
            CpgError::MissingField(format!("Node with id {:?} not found in left CPG", l_root))
        })?;

        let r_node = other.get_node_by_id(&r_root).ok_or_else(|| {
            CpgError::MissingField(format!("Node with id {:?} not found in right CPG", r_root))
        })?;

        if l_node != r_node {
            return Ok(vec![(
                Some(l_root),
                Some(r_root),
                format!(
                    "Node properties or type differ: left = {} {:?}, right = {} {:?}",
                    l_node.type_, l_node.properties, r_node.type_, r_node.properties
                ),
            )]);
        }

        let l_edges = self.get_deterministic_sorted_outgoing_edges(l_root);
        let r_edges = other.get_deterministic_sorted_outgoing_edges(r_root);

        let mut grouped_left: HashMap<(_, Vec<(_, _)>), Vec<_>> = HashMap::new();
        let mut grouped_right: HashMap<(_, Vec<(_, _)>), Vec<_>> = HashMap::new();

        for e in l_edges.iter() {
            grouped_left
                .entry((&e.type_, to_sorted_vec(&e.properties)))
                .or_default()
                .push(e);
        }
        for e in r_edges.iter() {
            grouped_right
                .entry((&e.type_, to_sorted_vec(&e.properties)))
                .or_default()
                .push(e);
        }

        let mut mismatches = Vec::new();
        for ((edge_type, props), left_group) in &grouped_left {
            let right_group = grouped_right.get(&(*edge_type, props.clone()));
            match right_group {
                Some(rg) => {
                    if **edge_type == EdgeType::SyntaxChild {
                        let ordered_left = self.ordered_syntax_children(l_root);
                        let ordered_right = other.ordered_syntax_children(r_root);

                        if ordered_left.len() != ordered_right.len() {
                            mismatches.push((
                                Some(l_root),
                                Some(r_root),
                                format!(
                                    "Ordered SyntaxChild count mismatch: #left = {}, #right = {}",
                                    ordered_left.len(),
                                    ordered_right.len()
                                ),
                            ));
                            return Ok(mismatches);
                        }

                        for (lc, rc) in ordered_left.iter().zip(ordered_right.iter()) {
                            mismatches.extend(self.compare_subtrees(other, *lc, *rc, visited)?);
                        }
                    } else {
                        if left_group.len() != rg.len() {
                            mismatches.push((Some(l_root), Some(r_root), format!(
                                "Edge count mismatch for type {:?} with props {:?}: #left = {}, #right = {}",
                                edge_type,
                                props,
                                left_group.len(),
                                rg.len()
                            )));
                            return Ok(mismatches);
                        }

                        for (l_edge, r_edge) in left_group.iter().zip(rg.iter()) {
                            mismatches.extend(
                                self.compare_subtrees(other, l_edge.to, r_edge.to, visited)?,
                            );
                        }
                    }
                }
                None => {
                    mismatches.push((
                        Some(l_root),
                        Some(r_root),
                        format!(
                            "Missing edge group in right CPG: type {:?} with props {:?}",
                            edge_type, props
                        ),
                    ));
                    return Ok(mismatches);
                }
            }
        }

        Ok(mismatches)
    }

    /// Format mismatch details with text diffs showing the actual source code differences
    fn source_diff(
        &self,
        other: &Cpg,
        mismatches: &Vec<(Option<NodeId>, Option<NodeId>, String)>,
    ) -> String {
        let mut details = vec![];

        for (left_node_opt, right_node_opt, detail) in mismatches {
            let left_source = left_node_opt
                .map(|node_id| self.get_node_source(&node_id))
                .unwrap_or_else(|| "".to_string());

            let right_source = right_node_opt
                .map(|node_id| other.get_node_source(&node_id))
                .unwrap_or_else(|| "".to_string());

            let diff = TextDiff::from_lines(&left_source, &right_source);

            // let mut diff_output = Vec::new();
            // for change in diff.iter_all_changes() {
            //     let sign = match change.tag() {
            //         ChangeTag::Delete => "- ",
            //         ChangeTag::Insert => "+ ",
            //         ChangeTag::Equal => "| ",
            //     };
            //     diff_output.push(format!("{}{}", sign, change.value().trim_end()));
            // }

            details.push(format!("Mismatch - {}:\n{}", detail, diff.unified_diff()));
        }

        details.join("\n")
    }

    /// Sexp Diff
    fn sexp_diff(&self, other: &Cpg, left_root: NodeId, right_root: NodeId) -> String {
        let left_sexp = self
            .serialize(&mut SexpSerializer::new(), Some(left_root))
            .unwrap_or("~BAD LEFT ROOT~".to_string());
        let right_sexp = other
            .serialize(&mut SexpSerializer::new(), Some(right_root))
            .unwrap_or("~BAD RIGHT ROOT~".to_string());

        let diff = TextDiff::from_lines(&left_sexp, &right_sexp);
        format!("{}", diff.unified_diff())
    }
}

#[cfg(test)]
mod tests {

    use crate::{
        cpg::{
            DescendantTraversal, DetailedComparisonResult, Edge, EdgeType, NodeType,
            tests::{create_test_cpg, create_test_node},
        },
        desc_trav,
    };
    use std::collections::HashMap;

    #[test]
    fn test_compare_equivalent_cpgs() {
        let mut cpg1 = create_test_cpg();
        let mut cpg2 = create_test_cpg();

        // Create identical structures
        for cpg in [&mut cpg1, &mut cpg2] {
            let root = cpg.add_node(create_test_node(NodeType::TranslationUnit, None), 0, 20);
            let func = cpg.add_node(
                create_test_node(
                    NodeType::Function {
                        name_traversals: vec![desc_trav![]],
                    },
                    Some("main".to_string()),
                ),
                1,
                19,
            );

            cpg.add_edge(Edge {
                from: root,
                to: func,
                type_: EdgeType::SyntaxChild,
                properties: HashMap::new(),
            });
        }

        let result = cpg1.compare(&cpg2).expect("Comparison failed");
        assert!(matches!(result, DetailedComparisonResult::Equivalent));
    }

    #[test]
    fn test_compare_different_functions() {
        let mut cpg1 = create_test_cpg();
        let mut cpg2 = create_test_cpg();

        // CPG1 has function "main"
        let root1 = cpg1.add_node(create_test_node(NodeType::TranslationUnit, None), 0, 20);
        let func1 = cpg1.add_node(
            create_test_node(
                NodeType::Function {
                    name_traversals: vec![desc_trav![]],
                },
                Some("main".to_string()),
            ),
            1,
            19,
        );
        cpg1.add_edge(Edge {
            from: root1,
            to: func1,
            type_: EdgeType::SyntaxChild,
            properties: HashMap::new(),
        });

        // CPG2 has function "test"
        let root2 = cpg2.add_node(create_test_node(NodeType::TranslationUnit, None), 0, 20);
        let func2 = cpg2.add_node(
            create_test_node(
                NodeType::Function {
                    name_traversals: vec![desc_trav![]],
                },
                Some("test".to_string()),
            ),
            1,
            19,
        );
        cpg2.add_edge(Edge {
            from: root2,
            to: func2,
            type_: EdgeType::SyntaxChild,
            properties: HashMap::new(),
        });

        let result = cpg1.compare(&cpg2).expect("Comparison failed");
        match result {
            DetailedComparisonResult::StructuralMismatch {
                only_in_left,
                only_in_right,
                ..
            } => {
                assert!(only_in_left.contains(&"main".to_string()));
                assert!(only_in_right.contains(&"test".to_string()));
            }
            _ => panic!("Expected structural mismatch"),
        }
    }

    #[test]
    fn test_compare_no_roots() {
        let cpg1 = create_test_cpg();
        let cpg2 = create_test_cpg();

        let result = cpg1.compare(&cpg2).expect("Comparison failed");
        assert!(matches!(result, DetailedComparisonResult::Equivalent));
    }

    #[test]
    fn test_compare_one_empty() {
        let cpg1 = create_test_cpg();
        let mut cpg2 = create_test_cpg();
        cpg2.add_node(create_test_node(NodeType::TranslationUnit, None), 0, 10);

        let result = cpg1.compare(&cpg2).expect("Comparison failed");
        match result {
            DetailedComparisonResult::StructuralMismatch { only_in_right, .. } => {
                assert!(only_in_right.contains(&"root".to_string()));
            }
            _ => panic!("Expected structural mismatch"),
        }
    }

    #[test]
    fn test_mre_bug_non_deterministic_ordered_syntax_children() {
        use std::collections::HashSet;

        let mut observed_orders = HashSet::new();

        // Run multiple times to potentially observe different orderings
        for run in 0..100 {
            let mut cpg = create_test_cpg();
            let root = cpg.add_node(create_test_node(NodeType::TranslationUnit, None), 0, 100);

            // Add multiple children in a specific order
            let child1 = cpg.add_node(create_test_node(NodeType::Statement, None), 10, 20);
            let child2 = cpg.add_node(create_test_node(NodeType::Statement, None), 30, 40);
            let child3 = cpg.add_node(create_test_node(NodeType::Statement, None), 50, 60);

            // Add parent-child edges (deliberately in different order than creation)
            cpg.add_edge(Edge {
                from: root,
                to: child2,
                type_: EdgeType::SyntaxChild,
                properties: HashMap::new(),
            });
            cpg.add_edge(Edge {
                from: root,
                to: child1,
                type_: EdgeType::SyntaxChild,
                properties: HashMap::new(),
            });
            cpg.add_edge(Edge {
                from: root,
                to: child3,
                type_: EdgeType::SyntaxChild,
                properties: HashMap::new(),
            });

            // No sibling edges - this creates ambiguity about which child comes first
            let ordered = cpg.ordered_syntax_children(root);
            observed_orders.insert(format!("{:?}", ordered));

            if observed_orders.len() > 1 {
                println!("Non-deterministic ordering detected on run {}", run);
                println!("Observed orders: {:?}", observed_orders);
                break;
            }
        }

        if observed_orders.len() > 1 {
            panic!(
                "Observed {} different orderings: {:?}",
                observed_orders.len(),
                observed_orders
            );
        } else {
            println!("No non-deterministic behavior observed in {} runs", 100);
        }
    }
}
