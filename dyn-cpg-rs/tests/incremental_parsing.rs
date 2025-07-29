use dyn_cpg_rs::{
    cpg::DetailedComparisonResult, diff::incremental_parse, languages::RegisteredLanguage,
    resource::Resource,
};
use tracing::debug;

/// Integration test for incremental parsing and CPG updates
/// This test verifies that incremental updates produce semantically equivalent CPGs
#[test]
fn test_incremental_reparse() {
    // dyn_cpg_rs::logging::init();
    debug!("Starting incremental reparse test");

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

    // Parse the new file incrementally
    let start = std::time::Instant::now();
    let (edits, new_tree) = incremental_parse(&mut parser, &old_src, &new_src, &mut old_tree)
        .expect("Failed to incrementally parse new file");
    debug!(
        "Incremental file parse took {} ms",
        start.elapsed().as_millis()
    );

    // Get changed ranges
    let changed_ranges = old_tree.changed_ranges(&new_tree);
    assert!(changed_ranges.len() != 0, "No changed ranges found");

    // Create CPG from original tree
    let start = std::time::Instant::now();
    let mut cpg = lang
        .cst_to_cpg(old_tree, new_src.clone())
        .expect("Failed to convert old tree to CPG");
    debug!(
        "Initial CPG creation took {} ms",
        start.elapsed().as_millis()
    );

    // Store metrics before incremental update
    let nodes_before = cpg.node_count();
    let edges_before = cpg.edge_count();

    // Perform the incremental update
    let start = std::time::Instant::now();
    cpg.incremental_update(edits, changed_ranges, &new_tree);
    debug!(
        "Incremental CPG update took {} ms",
        start.elapsed().as_millis()
    );

    // Compute the reference CPG from scratch
    let start = std::time::Instant::now();
    let new_cpg = lang
        .cst_to_cpg(new_tree, new_src)
        .expect("Failed to convert new tree to CPG");
    debug!(
        "Reference CPG creation took {} ms",
        start.elapsed().as_millis()
    );

    // Compare the incrementally updated CPG with the reference CPG
    std::fs::write("incr.dot", cpg.emit_dot()).expect("Failed to write incr.dot");
    std::fs::write("ref.dot", new_cpg.emit_dot()).expect("Failed to write ref.dot");
    let diff = cpg.compare(&new_cpg).expect("Failed to compare CPGs");
    match diff {
        DetailedComparisonResult::Equivalent => {}
        _ => panic!(
            "CPGs should be semantically equivalent, but found differences: {:?}",
            diff
        ),
    }

    // Verify the graph is still internally consistent
    assert!(cpg.node_count() > 0, "CPG should have nodes after update");
    assert!(cpg.edge_count() > 0, "CPG should have edges after update");
    assert!(
        cpg.get_root().is_some(),
        "CPG should have a root after update"
    );

    debug!(
        "Incremental update test passed: nodes {} -> {}, edges {} -> {}",
        nodes_before,
        cpg.node_count(),
        edges_before,
        cpg.edge_count()
    );
}

/// Integration test for incremental parsing and CPG updates
/// This test verifies that incremental updates produce semantically equivalent CPGs
#[test]
fn test_incremental_reparse_perf() {
    // Init the lang and parser
    let lang: RegisteredLanguage = "c".parse().expect("Failed to parse language");
    let mut parser = lang.get_parser().expect("Failed to get parser for C");

    // Load large_sample.old.c and large_sample.new.c from samples/
    let s_orig = Resource::new("samples/large_sample.old.c")
        .expect("Failed to create resource for large_sample.old.c");
    let s_new = Resource::new("samples/large_sample.new.c")
        .expect("Failed to create resource for large_sample.new.c");

    // Read the contents of the files
    let old_src = s_orig
        .read_bytes()
        .expect("Failed to read large_sample.old.c");
    let new_src = s_new
        .read_bytes()
        .expect("Failed to read large_sample.new.c");

    // Parse the original file
    let mut old_tree = parser
        .parse(old_src.clone(), None)
        .expect("Failed to parse original file");

    // Parse the new file incrementally
    let start = std::time::Instant::now();
    let (edits, new_tree) = incremental_parse(&mut parser, &old_src, &new_src, &mut old_tree)
        .expect("Failed to incrementally parse new file");
    println!(
        "Incremental file parse took {} ms",
        start.elapsed().as_millis()
    );

    // Get changed ranges
    let changed_ranges = old_tree.changed_ranges(&new_tree);
    assert!(changed_ranges.len() != 0, "No changed ranges found");

    // Create CPG from original tree
    let start = std::time::Instant::now();
    let mut cpg = lang
        .cst_to_cpg(old_tree, new_src.clone())
        .expect("Failed to convert old tree to CPG");
    println!(
        "Initial CPG creation took {} ms",
        start.elapsed().as_millis()
    );

    // Store metrics before incremental update
    let nodes_before = cpg.node_count();
    let edges_before = cpg.edge_count();

    // Perform the incremental update
    let start = std::time::Instant::now();
    cpg.incremental_update(edits, changed_ranges, &new_tree);
    println!(
        "Incremental CPG update took {} ms",
        start.elapsed().as_millis()
    );

    // Compute the reference CPG from scratch
    let start = std::time::Instant::now();
    let new_cpg = lang
        .cst_to_cpg(new_tree, new_src)
        .expect("Failed to convert new tree to CPG");
    println!(
        "Reference CPG creation took {} ms",
        start.elapsed().as_millis()
    );

    // Compare the incrementally updated CPG with the reference CPG
    let diff = cpg.compare(&new_cpg).expect("Failed to compare CPGs");
    match diff {
        DetailedComparisonResult::Equivalent => {}
        _ => panic!(
            "CPGs should be semantically equivalent, but found differences: {:?}",
            diff
        ),
    }

    // Verify the graph is still internally consistent
    assert!(cpg.node_count() > 0, "CPG should have nodes after update");
    assert!(cpg.edge_count() > 0, "CPG should have edges after update");
    assert!(
        cpg.get_root().is_some(),
        "CPG should have a root after update"
    );

    debug!(
        "Incremental update test passed: nodes {} -> {}, edges {} -> {}",
        nodes_before,
        cpg.node_count(),
        edges_before,
        cpg.edge_count()
    );
}

/// Test multiple sequential incremental updates
#[test]
fn test_multiple_incremental_updates() {
    let lang: RegisteredLanguage = "c".parse().expect("Failed to parse language");
    let mut parser = lang.get_parser().expect("Failed to get parser for C");

    // Start with a simple C program
    let source1 = b"int main() { return 0; }";
    let source2 = b"int main() { int x = 5; return x; }";
    let source3 = b"int main() { int x = 5; int y = 10; return x + y; }";

    // Parse initial source
    let mut tree = parser
        .parse(source1, None)
        .expect("Failed to parse initial source");

    let mut cpg = lang
        .cst_to_cpg(tree.clone(), source1.to_vec())
        .expect("Failed to create initial CPG");

    // First incremental update
    let (edits1, new_tree1) = incremental_parse(&mut parser, source1, source2, &mut tree)
        .expect("Failed to parse second source");

    let changed_ranges1 = tree.changed_ranges(&new_tree1);
    cpg.incremental_update(edits1, changed_ranges1, &new_tree1);
    tree = new_tree1;

    // Second incremental update
    let (edits2, new_tree2) = incremental_parse(&mut parser, source2, source3, &mut tree)
        .expect("Failed to parse third source");

    let changed_ranges2 = tree.changed_ranges(&new_tree2);
    cpg.incremental_update(edits2, changed_ranges2, &new_tree2);

    // Verify final result
    let reference_cpg = lang
        .cst_to_cpg(new_tree2, source3.to_vec())
        .expect("Failed to create reference CPG");

    let diff = cpg.compare(&reference_cpg).expect("Failed to compare CPGs");
    assert!(
        matches!(diff, DetailedComparisonResult::Equivalent),
        "Final CPG should match reference after multiple updates: {:?}",
        diff
    );
}

/// Test that incremental updates handle edge cases correctly
#[test]
fn test_incremental_edge_cases() {
    let lang: RegisteredLanguage = "c".parse().expect("Failed to parse language");
    let mut parser = lang.get_parser().expect("Failed to get parser for C");

    // Test empty to non-empty
    let empty_source = b"";
    let simple_source = b"int x;";

    if let Some(mut tree) = parser.parse(empty_source, None) {
        let mut cpg = lang
            .cst_to_cpg(tree.clone(), empty_source.to_vec())
            .expect("Failed to create CPG from empty source");

        let (edits, new_tree) =
            incremental_parse(&mut parser, empty_source, simple_source, &mut tree)
                .expect("Failed to parse simple source");

        let changed_ranges = tree.changed_ranges(&new_tree);
        cpg.incremental_update(edits, changed_ranges, &new_tree);

        let reference_cpg = lang
            .cst_to_cpg(new_tree, simple_source.to_vec())
            .expect("Failed to create reference CPG");

        let diff = cpg.compare(&reference_cpg).expect("Failed to compare CPGs");
        assert!(
            matches!(diff, DetailedComparisonResult::Equivalent),
            "Empty to non-empty should work: {:?}",
            diff
        );
    }
}
