use std::collections::HashMap;

use dyn_cpg_rs::{
    cpg::{
        Cpg, DescendantTraversal, DetailedComparisonResult, Edge, EdgeQuery, EdgeType, Node,
        NodeType,
    },
    desc_trav,
    diff::incremental_parse,
    languages::RegisteredLanguage,
    resource::Resource,
};

// --- Helper functions for tests --- //

fn create_test_cpg() -> Cpg {
    Cpg::new("C".parse().expect("Failed to parse language"), Vec::new())
}

fn create_test_node(node_type: NodeType) -> Node {
    Node {
        type_: node_type,
        properties: HashMap::new(),
    }
}

// --- Performance and stress tests --- //

#[test]
fn test_large_graph_operations() {
    let mut cpg = create_test_cpg();
    let mut nodes = Vec::new();

    // Create a larger graph structure
    for i in 0..1000 {
        let node = cpg.add_node(create_test_node(NodeType::Statement), i * 10, i * 10 + 9);
        nodes.push(node);
    }

    // Add edges between consecutive nodes
    for i in 0..999 {
        cpg.add_edge(Edge {
            from: nodes[i],
            to: nodes[i + 1],
            type_: EdgeType::ControlFlowEpsilon,
            properties: HashMap::new(),
        });
    }

    // Test spatial queries on large graph
    let overlapping = cpg.get_node_ids_by_offsets(500, 600);
    assert!(!overlapping.is_empty());

    // Test edge queries
    let all_control_flow = EdgeQuery::new()
        .edge_type(&EdgeType::ControlFlowEpsilon)
        .query(&cpg);
    assert_eq!(all_control_flow.len(), 999);

    // Test subtree removal on large graph
    let subtree_root = nodes[500];
    cpg.remove_subtree(subtree_root)
        .expect("Failed to remove subtree");
    assert!(cpg.get_node_by_id(&subtree_root).is_none());
}

// --- Original test with enhanced assertions --- //

#[test]
fn test_incr_reparse() {
    // Init the lang and parser
    let lang: RegisteredLanguage = "c".parse().expect("Failed to parse language");
    let mut parser = lang.get_parser().expect("Failed to get parser for C");

    // Load sample1.old.c and sample1.new.c from samples/
    let s_orig = Resource::new("samples/sample1.old.c")
        .expect("Failed to create resource for sample1.old.c");
    let s_new = Resource::new("samples/sample1.new.c")
        .expect("Failed to create resource for sample1.new.c");

    // Read the contents of the files
    let old_src = s_orig.read_bytes().expect("Failed to read sample1.old.c");
    let new_src = s_new.read_bytes().expect("Failed to read sample1.new.c");

    // Parse the original file
    let mut old_tree = parser
        .parse(old_src.clone(), None)
        .expect("Failed to parse original file");

    // Parse the new file
    let (edits, new_tree) = incremental_parse(&mut parser, &old_src, &new_src, &mut old_tree)
        .expect("Failed to incrementally parse new file");

    // Changed ranges
    let changed_ranges = old_tree.changed_ranges(&new_tree);
    assert!(changed_ranges.len() != 0, "No changed ranges found");

    let mut cpg = lang
        .cst_to_cpg(old_tree, new_src.clone())
        .expect("Failed to convert old tree to CPG");

    // Perform the incremental update
    cpg.incremental_update(edits, changed_ranges, &new_tree);

    // Compute the reference CPG
    let new_cpg = lang
        .cst_to_cpg(new_tree, new_src)
        .expect("Failed to convert new tree to CPG");

    // Check the difference between the two CPGs
    let diff = cpg.compare(&new_cpg).expect("Failed to compare CPGs");
    assert!(
        matches!(diff, DetailedComparisonResult::Equivalent),
        "CPGs should be semantically equivalent, but found differences: {:?}",
        diff
    );

    // Verify the graph is still internally consistent
    assert!(cpg.node_count() > 0, "CPG should have nodes after update");
    assert!(cpg.edge_count() > 0, "CPG should have edges after update");
    assert!(
        cpg.get_root().is_some(),
        "CPG should have a root after update"
    );
}

// --- Integration tests --- //

#[test]
fn test_multiple_incremental_updates() {
    // Test multiple sequential incremental updates
    let mut cpg = create_test_cpg();

    // Start with a simple structure
    let root = cpg.add_node(create_test_node(NodeType::TranslationUnit), 0, 100);
    let func = cpg.add_node(
        create_test_node(NodeType::Function {
            name_traversal: desc_trav![],
        }),
        10,
        90,
    );
    cpg.add_edge(Edge {
        from: root,
        to: func,
        type_: EdgeType::SyntaxChild,
        properties: HashMap::new(),
    });

    let initial_nodes = cpg.node_count();

    // Simulate multiple updates by removing and adding subtrees
    cpg.remove_subtree(func).expect("Failed to remove function");
    assert_eq!(cpg.node_count(), initial_nodes - 1);

    // Add it back with a different name
    let new_func = cpg.add_node(
        create_test_node(NodeType::Function {
            name_traversal: desc_trav![],
        }),
        10,
        90,
    );
    cpg.add_edge(Edge {
        from: root,
        to: new_func,
        type_: EdgeType::SyntaxChild,
        properties: HashMap::new(),
    });

    assert_eq!(cpg.node_count(), initial_nodes);

    // Verify structure is still valid
    let children = cpg.get_outgoing_edges(root);
    assert_eq!(children.len(), 1);
    assert_eq!(children[0].to, new_func);
}

#[test]
fn test_concurrent_modifications() {
    // Test that the CPG maintains consistency under various operation sequences
    let mut cpg = create_test_cpg();

    // Create a complex structure
    let root = cpg.add_node(create_test_node(NodeType::TranslationUnit), 0, 1000);
    let mut functions = Vec::new();

    for i in 0..5 {
        let func = cpg.add_node(
            create_test_node(NodeType::Function {
                name_traversal: desc_trav![],
            }),
            i * 200,
            (i + 1) * 200 - 1,
        );
        functions.push(func);
        cpg.add_edge(Edge {
            from: root,
            to: func,
            type_: EdgeType::SyntaxChild,
            properties: HashMap::new(),
        });
    }

    // Perform various operations
    let nodes_before = cpg.node_count();

    // Remove a function in the middle
    cpg.remove_subtree(functions[2])
        .expect("Failed to remove function");

    // Add a new function
    let new_func = cpg.add_node(
        create_test_node(NodeType::Function {
            name_traversal: desc_trav![],
        }),
        1001,
        1100,
    );
    cpg.add_edge(Edge {
        from: root,
        to: new_func,
        type_: EdgeType::SyntaxChild,
        properties: HashMap::new(),
    });

    // Verify consistency
    assert_eq!(cpg.node_count(), nodes_before); // One removed, one added
    assert!(cpg.get_node_by_id(&functions[2]).is_none());
    assert!(cpg.get_node_by_id(&new_func).is_some());

    // All remaining nodes should be reachable from root
    let children = cpg.get_outgoing_edges(root);
    assert_eq!(children.len(), 5); // 4 original + 1 new
}
