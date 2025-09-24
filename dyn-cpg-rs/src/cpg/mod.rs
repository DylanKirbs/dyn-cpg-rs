/// # The CPG (Code Property Graph) module
/// This module provides functionality to generate and update a Code Property Graph (CPG) from source code files.
/// As well as serialize and deserialize the CPG to and from a Gremlin database.
use crate::languages::RegisteredLanguage;
use slotmap::{SlotMap, new_key_type};
use std::collections::{HashMap, HashSet};
use strum_macros::Display;
use thiserror::Error;

// Public submodules
pub mod compare;
pub mod edge;
pub mod incremental;
pub mod node;
pub mod serialization;
pub use compare::*;
pub use edge::*;
pub use node::*;

// Private submodules
mod spatial_index;
use spatial_index::{BTreeIndex, SpatialIndex};

#[derive(Debug, Clone, Display, PartialEq, Eq, Hash)]
pub enum ListenerType {
    Unknown,
    // TODO: Figure out what we need here
}

// --- SlotMap Key Types --- //

new_key_type! {
    pub struct NodeId;
    pub struct EdgeId;
}

// --- Error handling for CPG operations --- //

#[derive(Debug, Display, Error)]
pub enum CpgError {
    InvalidFormat(String),
    MissingField(String),
    ConversionError(String),
    QueryExecutionError(String),
}

// --- Traversal Aid for Language Agnosticism --- //

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Represents a traversal path through the descendants of a node in the syntax tree.
/// Used to navigate from a subtree root, to a specific descendant node.
/// *This aids in the "language agnosticism" of the control flow pass.*
pub enum DescendantTraversal {
    Traversal(Vec<usize>),
    None,
}

impl DescendantTraversal {
    pub fn get_descendent(self, cpg: &Cpg, node: &NodeId) -> Option<NodeId> {
        let mut curr_node = *node;
        if let DescendantTraversal::Traversal(steps) = self {
            for step in steps {
                let children = cpg.ordered_syntax_children(curr_node);
                curr_node = *children.get(step)?;
            }
        } else {
            return None;
        }
        Some(curr_node)
    }
}

#[macro_export]
macro_rules! desc_trav {
    (None) => {
        DescendantTraversal::None
    };
    ($($step:expr),*) => {
        DescendantTraversal::Traversal(vec![$($step),*])
    };
}

// --- The graph structure for the CPG --- //

type Index = BTreeIndex;

#[derive(Debug, Clone)]
/// The Code Property Graph (CPG) structure
pub struct Cpg {
    /// The root node of the CPG, if set
    root: Option<NodeId>,
    /// Maps NodeId to Node
    nodes: SlotMap<NodeId, Node>,
    /// Maps EdgeId to Edge
    edges: SlotMap<EdgeId, Edge>,
    /// Maps NodeId to a list of EdgeIds that point to it
    incoming: HashMap<NodeId, HashSet<EdgeId>>,
    /// Maps NodeId to a list of EdgeIds that point from it
    outgoing: HashMap<NodeId, HashSet<EdgeId>>,
    /// Spatial index for fast lookups by byte range
    spatial_index: Index,
    /// The language of the CPG
    language: RegisteredLanguage,
    /// The source that the tree/CPG was parsed from
    source: Vec<u8>,
}

// Functionality to interact with the CPG

impl Cpg {
    pub fn new(language: RegisteredLanguage, source: Vec<u8>) -> Self {
        Cpg {
            nodes: SlotMap::with_key(),
            edges: SlotMap::with_key(),
            incoming: HashMap::new(),
            outgoing: HashMap::new(),
            spatial_index: Index::default(),
            root: None,
            language,
            source,
        }
    }

    pub fn set_root(&mut self, root: NodeId) {
        self.root = Some(root);
    }

    pub fn get_root(&self) -> Option<NodeId> {
        self.root
    }

    pub fn get_language(&self) -> &RegisteredLanguage {
        &self.language
    }

    pub fn get_source(&self) -> &Vec<u8> {
        &self.source
    }

    /// Update the source code for the CPG
    pub fn set_source(&mut self, source: Vec<u8>) {
        self.source = source;
    }

    /// Get the number of nodes in the CPG
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Add a node to the CPG and update the spatial index
    /// If no root is set, the first node added will be assumed to be the root, this can be overridden using `set_root`.
    pub fn add_node(&mut self, node: Node, start_byte: usize, end_byte: usize) -> NodeId {
        let node_id = self.nodes.insert(node);
        self.spatial_index.insert(node_id, start_byte, end_byte);

        if self.root.is_none() {
            self.set_root(node_id);
        }

        node_id
    }

    pub fn get_node_by_id(&self, id: &NodeId) -> Option<&Node> {
        self.nodes.get(*id)
    }

    pub fn get_node_by_id_mut(&mut self, id: &NodeId) -> Option<&mut Node> {
        self.nodes.get_mut(*id)
    }

    pub fn get_node_source(&self, node: &NodeId) -> String {
        let bytes: (usize, usize) = self.get_node_offsets_by_id(node).unwrap_or((0, 0));
        String::from_utf8_lossy(self.get_source().get(bytes.0..bytes.1).unwrap_or(&[])).to_string()
    }

    /// Get all of the Syntax Children, Grandchildren, etc. of a node, ordered by their SyntaxSibling edges (DFS)
    /// (i.e. in the order they appear in the source code)
    pub fn post_dfs_ordered_syntax_descendants(&self, root: NodeId) -> Vec<NodeId> {
        let mut result = Vec::new();
        let mut stack = vec![(root, false)];

        while let Some((node, visited)) = stack.pop() {
            if visited {
                result.push(node);
            } else {
                stack.push((node, true));
                let children = self.ordered_syntax_children(node);
                for &child in children.iter().rev() {
                    stack.push((child, false));
                }
            }
        }

        result
    }
}

// --- Tests --- //

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // --- Helper functions for tests --- //

    pub fn create_test_cpg() -> Cpg {
        Cpg::new("C".parse().expect("Failed to parse language"), Vec::new())
    }

    pub fn create_test_node(node_type: NodeType, name: Option<String>) -> Node {
        Node {
            type_: node_type.clone(),
            properties: {
                let mut prop = HashMap::new();
                if let Some(n) = name {
                    prop.insert("name".to_string(), n.clone());
                }
                prop
            },
        }
    }

    // --- Basic functionality tests --- //

    #[test]
    fn test_cpg_creation() {
        let cpg = Cpg::new("C".parse().expect("Failed to parse language"), Vec::new());
        assert!(cpg.nodes.is_empty());
        assert!(cpg.edges.is_empty());
        assert!(cpg.get_root().is_none());
    }

    #[test]
    fn test_add_node() {
        let mut cpg = create_test_cpg();
        let node = create_test_node(NodeType::TranslationUnit, None);
        let node_id = cpg.add_node(node.clone(), 0, 10);

        assert_eq!(cpg.get_node_by_id(&node_id), Some(&node));
        assert_eq!(cpg.get_root(), Some(node_id));
        assert_eq!(cpg.spatial_index.get_node_span(node_id), Some((0, 10)));
    }

    #[test]
    fn test_add_edge() {
        let mut cpg = create_test_cpg();
        let node_id1 = cpg.add_node(
            create_test_node(
                NodeType::Function {
                    name_traversals: vec![desc_trav!(None)],
                },
                Some("Test_func".to_string()),
            ),
            0,
            1,
        );
        let node_id2 = cpg.add_node(
            create_test_node(
                NodeType::Identifier {
                    type_: IdenType::UNKNOWN,
                },
                Some("x".to_string()),
            ),
            1,
            2,
        );

        let edge = Edge {
            from: node_id1,
            to: node_id2,
            type_: EdgeType::SyntaxChild,
            properties: HashMap::new(),
        };

        let edge_id = cpg.add_edge(edge.clone());
        assert_eq!(cpg.edges.get(edge_id), Some(&edge));

        // Test adjacency lists are updated
        let outgoing = cpg.get_outgoing_edges(node_id1);
        assert_eq!(outgoing.len(), 1);
        assert_eq!(outgoing[0], &edge);

        let incoming = cpg.get_incoming_edges(node_id2);
        assert_eq!(incoming.len(), 1);
        assert_eq!(incoming[0], &edge);
    }

    #[test]
    fn test_set_root_override() {
        let mut cpg = create_test_cpg();
        let node1 = cpg.add_node(
            create_test_node(
                NodeType::Function {
                    name_traversals: vec![desc_trav!(None)],
                },
                Some("test".to_string()),
            ),
            0,
            1,
        );
        let node2 = cpg.add_node(create_test_node(NodeType::TranslationUnit, None), 1, 2);

        assert_eq!(cpg.get_root(), Some(node1)); // First node becomes root

        cpg.set_root(node2);
        assert_eq!(cpg.get_root(), Some(node2)); // Root can be overridden
    }

    // --- Subtree removal tests --- //

    #[test]
    fn test_remove_subtree_simple() {
        let mut cpg = create_test_cpg();
        let root = cpg.add_node(
            create_test_node(
                NodeType::Function {
                    name_traversals: vec![desc_trav!(None)],
                },
                Some("main".to_string()),
            ),
            0,
            10,
        );
        let child1 = cpg.add_node(create_test_node(NodeType::Statement, None), 1, 5);
        let child2 = cpg.add_node(
            create_test_node(
                NodeType::Identifier {
                    type_: IdenType::UNKNOWN,
                },
                Some("x".to_string()),
            ),
            6,
            9,
        );

        cpg.add_edge(Edge {
            from: root,
            to: child1,
            type_: EdgeType::SyntaxChild,
            properties: HashMap::new(),
        });
        cpg.add_edge(Edge {
            from: root,
            to: child2,
            type_: EdgeType::SyntaxChild,
            properties: HashMap::new(),
        });

        let initial_node_count = cpg.nodes.len();
        let initial_edge_count = cpg.edges.len();

        cpg.remove_subtree(child1)
            .expect("Failed to remove subtree");

        // Child1 should be removed
        assert!(cpg.get_node_by_id(&child1).is_none());
        assert_eq!(cpg.nodes.len(), initial_node_count - 1);
        assert_eq!(cpg.edges.len(), initial_edge_count - 1);

        // Root and child2 should remain
        assert!(cpg.get_node_by_id(&root).is_some());
        assert!(cpg.get_node_by_id(&child2).is_some());
    }

    #[test]
    fn test_remove_subtree_recursive() {
        let mut cpg = create_test_cpg();
        let root = cpg.add_node(
            create_test_node(
                NodeType::Function {
                    name_traversals: vec![desc_trav!(None)],
                },
                Some("main".to_string()),
            ),
            0,
            20,
        );
        let child1 = cpg.add_node(create_test_node(NodeType::Block, None), 1, 10);
        let grandchild = cpg.add_node(create_test_node(NodeType::Statement, None), 2, 8);
        let child2 = cpg.add_node(
            create_test_node(
                NodeType::Identifier {
                    type_: IdenType::UNKNOWN,
                },
                None,
            ),
            11,
            19,
        );

        cpg.add_edge(Edge {
            from: root,
            to: child1,
            type_: EdgeType::SyntaxChild,
            properties: HashMap::new(),
        });
        cpg.add_edge(Edge {
            from: child1,
            to: grandchild,
            type_: EdgeType::SyntaxChild,
            properties: HashMap::new(),
        });
        cpg.add_edge(Edge {
            from: root,
            to: child2,
            type_: EdgeType::SyntaxChild,
            properties: HashMap::new(),
        });

        cpg.remove_subtree(child1)
            .expect("Failed to remove subtree");

        // Both child1 and grandchild should be removed
        assert!(cpg.get_node_by_id(&child1).is_none());
        assert!(cpg.get_node_by_id(&grandchild).is_none());

        // Root and child2 should remain
        assert!(cpg.get_node_by_id(&root).is_some());
        assert!(cpg.get_node_by_id(&child2).is_some());

        // No dangling edges should remain
        assert!(cpg.get_outgoing_edges(child1).is_empty());
        assert!(cpg.get_incoming_edges(child1).is_empty());
    }

    #[test]
    fn test_remove_subtree_edge_cleanup() {
        let mut cpg = create_test_cpg();
        let node1 = cpg.add_node(
            create_test_node(
                NodeType::Function {
                    name_traversals: vec![desc_trav!(None)],
                },
                Some("main".to_string()),
            ),
            0,
            5,
        );
        let node2 = cpg.add_node(create_test_node(NodeType::Statement, None), 6, 10);
        let node3 = cpg.add_node(
            create_test_node(
                NodeType::Identifier {
                    type_: IdenType::UNKNOWN,
                },
                None,
            ),
            11,
            15,
        );

        // Create a complex edge structure
        let edge1 = cpg.add_edge(Edge {
            from: node1,
            to: node2,
            type_: EdgeType::SyntaxChild,
            properties: HashMap::new(),
        });
        let edge2 = cpg.add_edge(Edge {
            from: node2,
            to: node3,
            type_: EdgeType::ControlFlowTrue,
            properties: HashMap::new(),
        });
        let edge3 = cpg.add_edge(Edge {
            from: node1,
            to: node3,
            type_: EdgeType::PDData("x".to_string()),
            properties: HashMap::new(),
        });

        // Remove node2, which should clean up associated edges
        cpg.remove_subtree(node2).expect("Failed to remove subtree");

        // Check that edges involving node2 are removed
        assert!(cpg.edges.get(edge1).is_none());
        assert!(cpg.edges.get(edge2).is_none());
        assert!(cpg.edges.get(edge3).is_some()); // This edge should remain

        // Check adjacency lists are updated
        assert!(cpg.get_outgoing_edges(node2).is_empty());
        assert!(cpg.get_incoming_edges(node2).is_empty());

        // Node1 should no longer have edge to node2
        let node1_outgoing = cpg.get_outgoing_edges(node1);
        assert_eq!(node1_outgoing.len(), 1);
        assert_eq!(node1_outgoing[0].to, node3);
    }

    // --- Ordered syntax children tests --- //

    #[test]
    fn test_ordered_syntax_children() {
        let mut cpg = create_test_cpg();
        let parent = cpg.add_node(
            create_test_node(
                NodeType::Function {
                    name_traversals: vec![desc_trav!(None)],
                },
                Some("main".to_string()),
            ),
            0,
            30,
        );
        let child1 = cpg.add_node(create_test_node(NodeType::Statement, None), 1, 10);
        let child2 = cpg.add_node(create_test_node(NodeType::Statement, None), 11, 20);
        let child3 = cpg.add_node(create_test_node(NodeType::Statement, None), 21, 29);

        // Add syntax child edges
        cpg.add_edge(Edge {
            from: parent,
            to: child1,
            type_: EdgeType::SyntaxChild,
            properties: HashMap::new(),
        });
        cpg.add_edge(Edge {
            from: parent,
            to: child2,
            type_: EdgeType::SyntaxChild,
            properties: HashMap::new(),
        });
        cpg.add_edge(Edge {
            from: parent,
            to: child3,
            type_: EdgeType::SyntaxChild,
            properties: HashMap::new(),
        });

        // Add sibling edges to establish order
        cpg.add_edge(Edge {
            from: child1,
            to: child2,
            type_: EdgeType::SyntaxSibling,
            properties: HashMap::new(),
        });
        cpg.add_edge(Edge {
            from: child2,
            to: child3,
            type_: EdgeType::SyntaxSibling,
            properties: HashMap::new(),
        });

        let ordered = cpg.ordered_syntax_children(parent);
        assert_eq!(ordered, vec![child1, child2, child3]);
    }

    #[test]
    fn test_ordered_syntax_children_empty() {
        let mut cpg = create_test_cpg();
        let parent = cpg.add_node(
            create_test_node(
                NodeType::Function {
                    name_traversals: vec![desc_trav!(None)],
                },
                Some("main".to_string()),
            ),
            0,
            10,
        );

        let ordered = cpg.ordered_syntax_children(parent);
        assert!(ordered.is_empty());
    }

    // --- Error handling tests --- //

    #[test]
    fn test_remove_nonexistent_subtree() {
        let mut cpg = create_test_cpg();
        let node = cpg.add_node(
            create_test_node(
                NodeType::Function {
                    name_traversals: vec![desc_trav!(None)],
                },
                Some("main".to_string()),
            ),
            0,
            10,
        );
        cpg.nodes.remove(node); // Manually remove to create invalid state

        // This should handle the case gracefully
        let result = cpg.remove_subtree(node);
        assert!(result.is_ok()); // Should not panic
    }

    // --- Property-based tests --- //
    // TODO: Move these to their respective submodules

    proptest! {
    #[test]
    fn prop_spatial_index_insertion_consistency(
        ranges in prop::collection::vec((0usize..1000, 0usize..1000), 1..50)
    ) {
        let mut cpg = create_test_cpg();
        let mut node_ids = Vec::new();

        // Insert nodes with generated ranges
        for (start, end) in ranges.iter() {
            let (start, end) = if start <= end { (*start, *end) } else { (*end, *start) };
            let node_id = cpg.add_node(create_test_node(NodeType::Statement, None), start, end);
            node_ids.push(node_id);
        }

        // Property: Every inserted node should be findable by its exact range
        // BUT: Zero-width ranges don't overlap with anything (including themselves)
        for (i, (start, end)) in ranges.iter().enumerate() {
            let (start, end) = if start <= end { (*start, *end) } else { (*end, *start) };
            let overlapping = cpg.spatial_index.get_nodes_covering_range(start, end);

            if start == end {
                // Zero-width ranges should NOT be found (they don't overlap with anything)
                prop_assert!(!overlapping.contains(&node_ids[i]),
                    "Zero-width range ({}, {}) should NOT be found in spatial index", start, end);
            } else {
                // Non-zero-width ranges should be found
                prop_assert!(overlapping.contains(&node_ids[i]),
                    "Node {} with range ({}, {}) should be found in spatial index", i, start, end);
            }
        }
    }

    #[test]
    fn prop_spatial_index_removal_consistency(
        ranges in prop::collection::vec((0usize..1000, 0usize..1000), 1..20)
    ) {
        let mut cpg = create_test_cpg();
        let mut node_ids = Vec::new();

        // Insert nodes
        for (start, end) in ranges.iter() {
            let (start, end) = if start <= end { (*start, *end) } else { (*end, *start) };
            let node_id = cpg.add_node(create_test_node(NodeType::Statement, None), start, end);
            node_ids.push(node_id);
        }

        // Remove every other node
        for (i, &node_id) in node_ids.iter().enumerate().step_by(2) {
            cpg.spatial_index.delete(node_id);

            // Property: Removed node should not be found in spatial index
            let (start, end) = ranges[i];
            let (start, end) = if start <= end { (start, end) } else { (end, start) };
            let overlapping = cpg.spatial_index.get_nodes_covering_range(start, end);
            prop_assert!(!overlapping.contains(&node_id),
                "Removed node {} should not be found in spatial index", i);
        }
    }

    #[test]
    fn prop_cpg_node_edge_consistency(
        node_count in 1usize..20,
        edge_pairs in prop::collection::vec((0usize..19, 0usize..19), 0..30)
    ) {
        let mut cpg = create_test_cpg();
        let mut node_ids = Vec::new();

        // Create nodes
        for i in 0..node_count {
            let node_id = cpg.add_node(
                create_test_node(NodeType::Statement, None),
                i * 10,
                (i + 1) * 10 - 1
            );
            node_ids.push(node_id);
        }

        // Add edges between valid node pairs
        for (from_idx, to_idx) in edge_pairs.iter() {
            if *from_idx < node_ids.len() && *to_idx < node_ids.len() && from_idx != to_idx {
                let edge = Edge {
                    from: node_ids[*from_idx],
                    to: node_ids[*to_idx],
                    type_: EdgeType::SyntaxChild,
                    properties: HashMap::new(),
                };
                cpg.add_edge(edge);
            }
        }

        // Property: Every edge should be found in both outgoing and incoming lists
        for edge in cpg.edges.values() {
            let outgoing = cpg.get_outgoing_edges(edge.from);
            let incoming = cpg.get_incoming_edges(edge.to);

            prop_assert!(outgoing.contains(&edge), "Edge should be in outgoing list");
            prop_assert!(incoming.contains(&edge), "Edge should be in incoming list");
        }
    }

    #[test]
    fn prop_cpg_comparison_reflexivity(
        node_types in prop::collection::vec(
            prop_oneof![
                Just(NodeType::Statement),
                Just(NodeType::Expression),
                Just(NodeType::Block),
                Just(NodeType::TranslationUnit),
            ],
            1..10
        )
    ) {
        let mut cpg = create_test_cpg();

        // Build a CPG with the generated node types
        for (i, node_type) in node_types.iter().enumerate() {
            cpg.add_node(
                create_test_node(node_type.clone(), None),
                i * 10,
                (i + 1) * 10 - 1
            );
        }

        // Property: A CPG should always be equivalent to itself
        let result = cpg.compare(&cpg).unwrap();
        prop_assert!(matches!(result, DetailedComparisonResult::Equivalent),
            "CPG self-comparison should always be equivalent");
    }

    #[test]
    fn prop_subtree_removal_maintains_consistency(
        initial_nodes in 3usize..15,
        removal_indices in prop::collection::vec(0usize..14, 1..5)
    ) {
        let mut cpg = create_test_cpg();
        let mut node_ids = Vec::new();

        // Create initial nodes
        for i in 0..initial_nodes {
            let node_id = cpg.add_node(
                create_test_node(NodeType::Statement, None),
                i * 100,
                (i + 1) * 100 - 1
            );
            node_ids.push(node_id);
        }

        // Add some parent-child relationships
        for i in 1..initial_nodes {
            if i > 0 {
                cpg.add_edge(Edge {
                    from: node_ids[i - 1],
                    to: node_ids[i],
                    type_: EdgeType::SyntaxChild,
                    properties: HashMap::new(),
                });
            }
        }

        let initial_edge_count = cpg.edge_count();

        // Remove nodes at valid indices
        let mut _removed_count = 0;
        for &idx in removal_indices.iter() {
            if idx < node_ids.len() && cpg.nodes.contains_key(node_ids[idx]) {
                                        cpg.remove_subtree(node_ids[idx]).unwrap();
                _removed_count += 1;
            }
        }

        // Property: After removal, no dangling edges should exist
        for edge in cpg.edges.values() {
            prop_assert!(cpg.nodes.contains_key(edge.from),
                "Edge 'from' node should exist after subtree removal");
            prop_assert!(cpg.nodes.contains_key(edge.to),
    "Edge 'to' node should exist after subtree removal");
        }

        // Property: Edge count should decrease (or stay same if no edges were removed)
        prop_assert!(cpg.edge_count() <= initial_edge_count,
            "Edge count should not increase after subtree removal");
    }

    #[test]
    fn prop_ordered_syntax_children_consistency(
        child_count in  1usize..10
    ) {
        let mut cpg = create_test_cpg();
        let parent = cpg.add_node(
            create_test_node(NodeType::Function {
                name_traversals: vec![desc_trav!(None)],
            },
            Some("test_func".to_string()),
        ),
            0,
            child_count * 100
        );

        let mut children = Vec::new();
        for i in 0..child_count {
            let child = cpg.add_node(
                create_test_node(NodeType::Statement, None),
                i * 10,
                (i + 1) * 10 - 1
            );
            children.push(child);

            // Add parent-child edge
            cpg.add_edge(Edge {
                from: parent,
                to: child,
                type_: EdgeType::SyntaxChild,
                properties: HashMap::new(),
            });
        }

        // Add sibling edges to establish order
        for i in 0..(child_count - 1) {
            cpg.add_edge(Edge {
                from: children[i],
                to: children[i + 1],
                type_: EdgeType::SyntaxSibling,
                properties: HashMap::new(),
            });
        }

        let ordered = cpg.ordered_syntax_children(parent);

        // Property: Ordered children should contain all children exactly once
        prop_assert_eq!(ordered.len(), child_count, "Should have all children");

        // Property: Order should match the sibling chain
        for i in 0..(child_count - 1) {
            let current_idx = ordered.iter().position(|&x| x == children[i]).unwrap();
            let next_idx = ordered.iter().position(|&x| x == children[i + 1]).unwrap();
            prop_assert!(next_idx == current_idx + 1,
                "Sibling order should be preserved: child {} should come before child {}", i, i + 1);
        }
    }
    }
}
