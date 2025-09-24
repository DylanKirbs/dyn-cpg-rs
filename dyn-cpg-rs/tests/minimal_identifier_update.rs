use dyn_cpg_rs::diff::incremental_ts_parse;
use dyn_cpg_rs::languages::RegisteredLanguage;

/// Minimal test to isolate the identifier name update issue
///
/// Problem: When doing incremental updates, identifier nodes inside functions
/// don't get their "name" property updated properly during surgical updates.
///
/// Expected: Incremental CPG should have identifier with name "d"
/// Actual: Incremental CPG has identifier with name "_" (truncated from original)
#[test]
fn test_identifier_name_update_isolated() {
    dyn_cpg_rs::logging::init();
    let lang: RegisteredLanguage = "c".parse().expect("Failed to parse language");
    let mut parser = lang.get_parser().expect("Failed to get parser for C");

    // Simple function name change - this should trigger surgical update on the identifier
    let old_source = b"int _m___c_A0() { return 0; }";
    let new_source = b"int d() { return 1; }";

    println!("=== SOURCE TRANSFORMATION ===");
    println!("Old: {}", std::str::from_utf8(old_source).unwrap());
    println!("New: {}", std::str::from_utf8(new_source).unwrap());

    // Parse old source and create initial CPG
    let old_tree = parser
        .parse(old_source, None)
        .expect("Old source should parse");
    let mut incremental_cpg = lang
        .cst_to_cpg(old_tree.clone(), old_source.to_vec())
        .expect("Failed to create CPG from old source");

    println!("\n=== INITIAL CPG ANALYSIS ===");
    if let Some(root_id) = incremental_cpg.get_root() {
        let mut all_nodes = incremental_cpg.post_dfs_ordered_syntax_descendants(root_id);
        all_nodes.push(root_id); // Include root itself
        for node_id in &all_nodes {
            if let Some(node) = incremental_cpg.get_node_by_id(node_id) {
                if let Some(name) = node.properties.get("name") {
                    println!("Node {:?}: type={:?}, name={:?}", node_id, node.type_, name);
                }
            }
        }
    }

    // Parse new source for reference
    let new_tree = parser
        .parse(new_source, None)
        .expect("New source should parse");
    let reference_cpg = lang
        .cst_to_cpg(new_tree.clone(), new_source.to_vec())
        .expect("Failed to create reference CPG");

    println!("\n=== REFERENCE CPG ANALYSIS ===");
    if let Some(root_id) = reference_cpg.get_root() {
        let mut all_nodes = reference_cpg.post_dfs_ordered_syntax_descendants(root_id);
        all_nodes.push(root_id); // Include root itself
        for node_id in &all_nodes {
            if let Some(node) = reference_cpg.get_node_by_id(node_id) {
                if let Some(name) = node.properties.get("name") {
                    println!("Node {:?}: type={:?}, name={:?}", node_id, node.type_, name);
                }
            }
        }
    }

    // Do incremental parsing
    let mut old_tree_copy = old_tree.clone();
    let (edits, new_tree) =
        incremental_ts_parse(&mut parser, old_source, new_source, &mut old_tree_copy)
            .expect("Incremental parse should succeed");

    println!("\n=== INCREMENTAL UPDATE ===");
    println!("Edits: {:?}", edits);

    let changed_ranges: Vec<_> = old_tree.changed_ranges(&new_tree).collect();
    println!("Changed ranges: {:?}", changed_ranges);

    // Apply incremental update
    incremental_cpg
        .incremental_update(&mut parser, new_source.to_vec())
        .expect("Incremental update should succeed");

    println!("\n=== POST-UPDATE CPG ANALYSIS ===");
    if let Some(root_id) = incremental_cpg.get_root() {
        let mut all_nodes = incremental_cpg.post_dfs_ordered_syntax_descendants(root_id);
        all_nodes.push(root_id); // Include root itself
        for node_id in &all_nodes {
            if let Some(node) = incremental_cpg.get_node_by_id(node_id) {
                if let Some(name) = node.properties.get("name") {
                    let source_text = incremental_cpg.get_node_source(node_id);
                    println!(
                        "Node {:?}: type={:?}, name={:?}, source_text={:?}",
                        node_id, node.type_, name, source_text
                    );
                }
            }
        }
    }

    // Find the function identifier in both CPGs
    let find_function_identifier = |cpg: &dyn_cpg_rs::cpg::Cpg| -> Option<String> {
        if let Some(root_id) = cpg.get_root() {
            let mut all_nodes = cpg.post_dfs_ordered_syntax_descendants(root_id);
            all_nodes.push(root_id);
            for node_id in &all_nodes {
                if let Some(node) = cpg.get_node_by_id(node_id) {
                    if matches!(
                        node.type_,
                        dyn_cpg_rs::cpg::node::NodeType::Identifier { .. }
                    ) {
                        if let Some(name) = node.properties.get("name") {
                            // Look for the identifier that's likely the function name
                            if name != "return" {
                                // Filter out non-function identifiers
                                return Some(name.clone());
                            }
                        }
                    }
                }
            }
        }
        None
    };

    let incremental_function_name = find_function_identifier(&incremental_cpg);
    let reference_function_name = find_function_identifier(&reference_cpg);

    println!("\n=== COMPARISON ===");
    println!(
        "Incremental function identifier: {:?}",
        incremental_function_name
    );
    println!(
        "Reference function identifier: {:?}",
        reference_function_name
    );

    // This is the core assertion that should pass but currently fails
    assert_eq!(
        incremental_function_name, reference_function_name,
        "Function identifier names should match after incremental update"
    );
}
