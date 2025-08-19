use super::{Cpg, CpgError, NodeId, edge::EdgeType, node::NodeType};
use similar::TextDiff;
use std::collections::{HashMap, HashSet};
use tracing::debug;

// --- Comparison Results --- //

/// Detailed result of a CPG comparison
#[derive(Debug, Clone, PartialEq)]
pub enum DetailedComparisonResult {
    /// The CPGs are semantically equivalent
    Equivalent,
    /// The CPGs have structural differences
    StructuralMismatch {
        /// Functions present only in the left CPG
        only_in_left: Vec<String>,
        /// Functions present only in the right CPG
        only_in_right: Vec<String>,
        /// Functions that exist in both but have differences
        function_mismatches: Vec<FunctionComparisonResult>,
    },
}

/// Result of comparing a single function between two CPGs
#[derive(Debug, Clone, PartialEq)]
pub enum FunctionComparisonResult {
    /// The functions are equivalent
    Equivalent,
    /// The functions differ
    Mismatch {
        /// The name of the function
        function_name: String,
        /// Details about the mismatch
        details: String,
    },
}

// --- Helper Functions --- //

fn unescape_string(input: &str) -> String {
    let mut result = String::new();
    let mut chars = input.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '\\' {
            if let Some(&next_ch) = chars.peek() {
                match next_ch {
                    'n' => {
                        chars.next();
                        result.push('\n');
                    }
                    't' => {
                        chars.next();
                        result.push('\t');
                    }
                    'r' => {
                        chars.next();
                        result.push('\r');
                    }
                    '\\' => {
                        chars.next();
                        result.push('\\');
                    }
                    '\'' => {
                        chars.next();
                        result.push('\'');
                    }
                    '\"' => {
                        chars.next();
                        result.push('\"');
                    }
                    '0' => {
                        chars.next();
                        result.push('\0');
                    }
                    'x' => {
                        chars.next();
                        let mut hex_chars = String::new();
                        for _ in 0..2 {
                            if let Some(&hex_ch) = chars.peek() {
                                if hex_ch.is_ascii_hexdigit() {
                                    hex_chars.push(chars.next().unwrap());
                                } else {
                                    break;
                                }
                            }
                        }
                        if let Ok(hex_val) = u8::from_str_radix(&hex_chars, 16) {
                            if hex_val.is_ascii() {
                                result.push(hex_val as char);
                            } else {
                                result.push('\\');
                                result.push('x');
                                result.push_str(&hex_chars);
                            }
                        } else {
                            result.push('\\');
                            result.push('x');
                            result.push_str(&hex_chars);
                        }
                    }
                    'u' => {
                        chars.next();
                        if chars.peek() == Some(&'{') {
                            chars.next();
                            let mut unicode_chars = String::new();
                            while let Some(&unicode_ch) = chars.peek() {
                                if unicode_ch == '}' {
                                    chars.next();
                                    break;
                                } else if unicode_ch.is_ascii_hexdigit() {
                                    unicode_chars.push(chars.next().unwrap());
                                } else {
                                    break;
                                }
                            }
                            if let Ok(unicode_val) = u32::from_str_radix(&unicode_chars, 16) {
                                if let Some(unicode_char) = char::from_u32(unicode_val) {
                                    result.push(unicode_char);
                                } else {
                                    result.push_str(&format!("\\u{{{}}}", unicode_chars));
                                }
                            } else {
                                result.push_str(&format!("\\u{{{}}}", unicode_chars));
                            }
                        } else {
                            result.push('\\');
                            result.push('u');
                        }
                    }
                    _ => {
                        result.push(ch);
                        result.push(next_ch);
                        chars.next();
                    }
                }
            } else {
                result.push(ch);
            }
        } else {
            result.push(ch);
        }
    }

    result
}

fn to_sorted_vec(properties: &HashMap<String, String>) -> Vec<(String, String)> {
    let mut vec: Vec<_> = properties
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    vec.sort_by(|a, b| a.0.cmp(&b.0));
    vec
}

// --- CPG Comparison Implementation --- //

impl Cpg {
    /// Compare two CPGs for semantic equality
    /// Returns a detailed comparison result indicating structural differences and function-level mismatches
    pub fn compare(&self, other: &Cpg) -> Result<DetailedComparisonResult, CpgError> {
        debug!(
            "Comparing CPGs: left root = {:?}, right root = {:?}",
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
                    let mut mismatches = Vec::new();
                    let mut visited = std::collections::HashSet::new();
                    self.compare_subtrees(other, &mut mismatches, l_root, r_root, &mut visited)?;
                    if mismatches.is_empty() {
                        return Ok(DetailedComparisonResult::Equivalent);
                    } else {
                        return Ok(DetailedComparisonResult::StructuralMismatch {
                            only_in_left: vec![],
                            only_in_right: vec![],
                            function_mismatches: vec![FunctionComparisonResult::Mismatch {
                                function_name: "root".to_string(),
                                details: self.format_mismatch_details_with_diff(
                                    other,
                                    &format!(
                                        "Root nodes differ [Root not TranslationUnit (left={},right={})]",
                                        l_node.type_ != NodeType::TranslationUnit,
                                        r_node.type_ != NodeType::TranslationUnit,
                                    ),
                                    &mismatches,
                                )
                            }],
                        });
                    }
                }

                // Compare top-level structure
                let left_functions = self.get_top_level_functions(l_root)?;
                let right_functions = other.get_top_level_functions(r_root)?;

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
                        let mut mismatches = Vec::new();
                        let mut visited = std::collections::HashSet::new();
                        self.compare_subtrees(
                            other,
                            &mut mismatches,
                            *left_func_id,
                            *right_func_id,
                            &mut visited,
                        )?;

                        if !mismatches.is_empty() {
                            function_mismatches.push(FunctionComparisonResult::Mismatch {
                                function_name: name.clone(),
                                details: self.format_mismatch_details_with_diff(
                                    other,
                                    &format!("Function {} has structural differences", name),
                                    &mismatches,
                                ),
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
        let child_edges = self.get_outgoing_edges(root);

        debug!(
            "Looking for functions in {} child edges from root {:?}",
            child_edges.len(),
            root
        );

        for edge in child_edges {
            if edge.type_ == EdgeType::SyntaxChild {
                let node = self.get_node_by_id(&edge.to).ok_or_else(|| {
                    CpgError::MissingField(format!("Child node with id {:?} not found", edge.to))
                })?;

                debug!(
                    "Child node type: {:?}, properties: {:?}",
                    node.type_, node.properties
                );

                // Check if this child is a Function node
                if let NodeType::Function { .. } = node.type_ {
                    // Try to get the function name from properties
                    let name = node
                        .properties
                        .get("name")
                        .cloned()
                        .unwrap_or_else(|| format!("unnamed_function_{:?}", edge.to));
                    debug!("Found function with name: {}", name);
                    functions.insert(name, edge.to);
                }
            }
        }

        debug!(
            "Found {} functions: {:?}",
            functions.len(),
            functions.keys().collect::<Vec<_>>()
        );
        Ok(functions)
    }

    /// Compare two subtrees, updating a list of the NodeIds of the sub-subtrees that are mismatched
    fn compare_subtrees(
        &self,
        other: &Cpg,
        mismatches: &mut Vec<(Option<NodeId>, Option<NodeId>)>,
        l_root: NodeId,
        r_root: NodeId,
        visited: &mut HashSet<(NodeId, NodeId)>,
    ) -> Result<(), CpgError> {
        // Avoid re-comparing the same pair of nodes
        if !visited.insert((l_root, r_root)) {
            return Ok(());
        }

        let l_node = self.get_node_by_id(&l_root).ok_or_else(|| {
            CpgError::MissingField(format!("Node with id {:?} not found in left CPG", l_root))
        })?;

        let r_node = other.get_node_by_id(&r_root).ok_or_else(|| {
            CpgError::MissingField(format!("Node with id {:?} not found in right CPG", r_root))
        })?;

        if l_node != r_node {
            mismatches.push((Some(l_root), Some(r_root)));
            return Ok(());
        }

        let l_edges = self.get_outgoing_edges(l_root);
        let r_edges = other.get_outgoing_edges(r_root);

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

        for ((edge_type, props), left_group) in &grouped_left {
            let right_group = grouped_right.get(&(*edge_type, props.clone()));
            match right_group {
                Some(rg) => {
                    if **edge_type == EdgeType::SyntaxChild {
                        let ordered_left = self.ordered_syntax_children(l_root);
                        let ordered_right = other.ordered_syntax_children(r_root);

                        if ordered_left.len() != ordered_right.len() {
                            mismatches.push((Some(l_root), Some(r_root)));
                            return Ok(());
                        }

                        for (lc, rc) in ordered_left.iter().zip(ordered_right.iter()) {
                            self.compare_subtrees(other, mismatches, *lc, *rc, visited)?; // Pass visited
                        }
                    } else {
                        if left_group.len() != rg.len() {
                            mismatches.push((Some(l_root), Some(r_root)));
                            return Ok(());
                        }

                        for (l_edge, r_edge) in left_group.iter().zip(rg.iter()) {
                            self.compare_subtrees(
                                other, mismatches, l_edge.to, r_edge.to, visited,
                            )?; // Pass visited
                        }
                    }
                }
                None => {
                    mismatches.push((Some(l_root), Some(r_root)));
                    return Ok(());
                }
            }
        }

        Ok(())
    }

    /// Format mismatch details with text diffs showing the actual source code differences
    fn format_mismatch_details_with_diff(
        &self,
        other: &Cpg,
        base_message: &str,
        mismatches: &Vec<(Option<NodeId>, Option<NodeId>)>,
    ) -> String {
        let mut details = vec![base_message.to_string()];

        for (left_node_opt, right_node_opt) in mismatches {
            let left_source = left_node_opt
                .map(|node_id| self.get_node_source(&node_id))
                .unwrap_or_else(|| "".to_string());

            let right_source = right_node_opt
                .map(|node_id| other.get_node_source(&node_id))
                .unwrap_or_else(|| "".to_string());

            let left_unescaped = unescape_string(&left_source);
            let right_unescaped = unescape_string(&right_source);

            let diff = TextDiff::from_lines(&left_unescaped, &right_unescaped);

            // let mut diff_output = Vec::new();
            // for change in diff.iter_all_changes() {
            //     let sign = match change.tag() {
            //         ChangeTag::Delete => "- ",
            //         ChangeTag::Insert => "+ ",
            //         ChangeTag::Equal => "| ",
            //     };
            //     diff_output.push(format!("{}{}", sign, change.value().trim_end()));
            // }

            details.push(format!(
                "\nMismatch:\n{}",
                // diff_output.join("\n")
                diff.unified_diff()
            ));
        }

        details.join("\n")
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
            let root = cpg.add_node(create_test_node(NodeType::TranslationUnit), 0, 20);
            let func = cpg.add_node(
                create_test_node(NodeType::Function {
                    name_traversal: desc_trav![],
                    name: Some("main".to_string()),
                }),
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
        let root1 = cpg1.add_node(create_test_node(NodeType::TranslationUnit), 0, 20);
        let func1 = cpg1.add_node(
            create_test_node(NodeType::Function {
                name_traversal: desc_trav![],
                name: Some("main".to_string()),
            }),
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
        let root2 = cpg2.add_node(create_test_node(NodeType::TranslationUnit), 0, 20);
        let func2 = cpg2.add_node(
            create_test_node(NodeType::Function {
                name_traversal: desc_trav![],
                name: Some("test".to_string()),
            }),
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
        cpg2.add_node(create_test_node(NodeType::TranslationUnit), 0, 10);

        let result = cpg1.compare(&cpg2).expect("Comparison failed");
        match result {
            DetailedComparisonResult::StructuralMismatch { only_in_right, .. } => {
                assert!(only_in_right.contains(&"root".to_string()));
            }
            _ => panic!("Expected structural mismatch"),
        }
    }
}
