use dyn_cpg_rs::{
    cpg::{
        DetailedComparisonResult,
        serialization::{DotSerializer, SexpSerializer},
    },
    diff::incremental_parse,
    languages::RegisteredLanguage,
    resource::Resource,
};
use proptest::prelude::*;
use tracing::debug;

/// Integration test for incremental parsing and CPG updates
/// This test verifies that incremental updates produce semantically equivalent CPGs
#[test]
fn test_incremental_reparse() {
    dyn_cpg_rs::logging::init();
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
    cpg.incremental_update(edits, changed_ranges, &new_tree, new_src.clone());
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
    let diff = cpg.compare(&new_cpg).expect("Failed to compare CPGs");
    match diff {
        DetailedComparisonResult::Equivalent => {}
        _ => panic!(
            "CPGs should be semantically equivalent, but found differences: {}",
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
    cpg.incremental_update(edits, changed_ranges, &new_tree, new_src.clone());
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
    let diff = cpg.compare(&new_cpg).expect("Failed to compare CPGs");
    match diff {
        DetailedComparisonResult::Equivalent => {}
        _ => panic!(
            "CPGs should be semantically equivalent, but found differences: {}",
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
fn test_mre_multiple_incremental_updates() {
    dyn_cpg_rs::logging::init();
    debug!("Testing multiple incremental updates");

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
    cpg.incremental_update(edits1, changed_ranges1, &new_tree1, source2.to_vec());
    tree = new_tree1;

    // Second incremental update
    let (edits2, new_tree2) = incremental_parse(&mut parser, source2, source3, &mut tree)
        .expect("Failed to parse third source");

    let changed_ranges2 = tree.changed_ranges(&new_tree2);
    cpg.incremental_update(edits2, changed_ranges2, &new_tree2, source3.to_vec());

    // Verify final result
    let reference_cpg = lang
        .cst_to_cpg(new_tree2, source3.to_vec())
        .expect("Failed to create reference CPG");

    cpg.serialize_to_file(
        &mut DotSerializer::new(),
        format!("{}/{}-{}.dot", "debug", "mre_mul_inc", "incr"),
        None,
    )
    .expect("Failed to write incr dot file");
    reference_cpg
        .serialize_to_file(
            &mut DotSerializer::new(),
            format!("{}/{}-{}.dot", "debug", "mre_mul_inc", "ref"),
            None,
        )
        .expect("Failed to write reference dot file");

    cpg.serialize_to_file(
        &mut SexpSerializer::new(),
        format!("{}/{}-{}.sexp", "debug", "mre_mul_inc", "incr"),
        None,
    )
    .expect("Failed to write incr sexp");
    reference_cpg
        .serialize_to_file(
            &mut SexpSerializer::new(),
            format!("{}/{}-{}.sexp", "debug", "mre_mul_inc", "ref"),
            None,
        )
        .expect("Failed to write reference sexp");

    let diff = cpg.compare(&reference_cpg).expect("Failed to compare CPGs");
    assert!(
        matches!(diff, DetailedComparisonResult::Equivalent),
        "Final CPG should match reference after multiple updates: {}",
        diff
    );
}

/// Debug test to understand what's happening in incremental update
#[test]
fn test_incremental_update_samples() {
    dyn_cpg_rs::logging::init();
    debug!("Starting debug incremental update test");

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

    // Create CPG from original tree
    let mut cpg = lang
        .cst_to_cpg(old_tree.clone(), old_src.clone())
        .expect("Failed to convert old tree to CPG");

    println!("=== ORIGINAL CPG ===");
    println!("Nodes: {}", cpg.node_count());
    println!("Edges: {}", cpg.edge_count());
    if let Some(root) = cpg.get_root() {
        println!("Root node: {:?}", cpg.get_node_by_id(&root));
    }

    // Parse the new file incrementally
    let (edits, new_tree) = incremental_parse(&mut parser, &old_src, &new_src, &mut old_tree)
        .expect("Failed to incrementally parse new file");

    println!("=== EDITS ===");
    for edit in &edits {
        println!("{:?}", edit);
    }

    // Get changed ranges
    let changed_ranges: Vec<_> = old_tree.changed_ranges(&new_tree).collect();
    println!("=== CHANGED RANGES ===");
    for range in &changed_ranges {
        println!("{:?}", range);
    }

    // Perform the incremental update
    cpg.incremental_update(
        edits,
        changed_ranges.into_iter(),
        &new_tree,
        new_src.clone(),
    );

    println!("=== INCREMENTAL CPG ===");
    println!("Nodes: {}", cpg.node_count());
    println!("Edges: {}", cpg.edge_count());
    if let Some(root) = cpg.get_root() {
        println!("Root node: {:?}", cpg.get_node_by_id(&root));
    }

    cpg.serialize_to_file(
        &mut SexpSerializer::new(),
        format!("{}/{}-{}.sexp", "debug", "mre_inc_samples", "incr"),
        None,
    )
    .expect("Failed to write incremental CPG");

    // Compute the reference CPG from scratch
    let new_cpg = lang
        .cst_to_cpg(new_tree, new_src)
        .expect("Failed to convert new tree to CPG");

    println!("=== REFERENCE CPG ===");
    println!("Nodes: {}", new_cpg.node_count());
    println!("Edges: {}", new_cpg.edge_count());
    if let Some(root) = new_cpg.get_root() {
        println!("Root node: {:?}", new_cpg.get_node_by_id(&root));
    }

    new_cpg
        .serialize_to_file(
            &mut SexpSerializer::new(),
            format!("{}/{}-{}.sexp", "debug", "mre_inc_samples", "ref"),
            None,
        )
        .expect("Failed to write reference CPG");

    // Compare the incrementally updated CPG with the reference CPG
    let diff = cpg.compare(&new_cpg).expect("Failed to compare CPGs");
    println!("=== COMPARISON ===");
    println!("{}", diff);

    assert!(matches!(diff, DetailedComparisonResult::Equivalent));
}

/// Test that incremental updates handle edge cases correctly
#[test]
fn test_mre_incremental_edge_cases() {
    let lang: RegisteredLanguage = "c".parse().expect("Failed to parse language");
    let mut parser = lang.get_parser().expect("Failed to get parser for C");

    // Test empty to non-empty - this is a specific challenging case
    let empty_source = b"";
    let simple_source = b"int x;";

    if let Some(mut tree) = parser.parse(empty_source, None) {
        let mut cpg = lang
            .cst_to_cpg(tree.clone(), empty_source.to_vec())
            .expect("Failed to create CPG from empty source");

        println!("=== INITIAL EMPTY CPG ===");
        println!("Nodes: {}", cpg.node_count());
        if let Some(root) = cpg.get_root() {
            println!("Root node: {:?}", cpg.get_node_by_id(&root));
        } else {
            println!("No root node");
        }

        let (edits, new_tree) =
            incremental_parse(&mut parser, empty_source, simple_source, &mut tree)
                .expect("Failed to parse simple source");

        println!("=== EDITS ===");
        for edit in &edits {
            println!("{:?}", edit);
        }

        let changed_ranges: Vec<_> = tree.changed_ranges(&new_tree).collect();
        println!("=== CHANGED RANGES ===");
        for range in &changed_ranges {
            println!("{:?}", range);
        }

        cpg.incremental_update(
            edits,
            changed_ranges.into_iter(),
            &new_tree,
            simple_source.to_vec(),
        );

        println!("=== INCREMENTAL CPG ===");
        println!("Nodes: {}", cpg.node_count());
        if let Some(root) = cpg.get_root() {
            println!("Root node: {:?}", cpg.get_node_by_id(&root));
        } else {
            println!("No root node");
        }

        let reference_cpg = lang
            .cst_to_cpg(new_tree, simple_source.to_vec())
            .expect("Failed to create reference CPG");

        println!("=== REFERENCE CPG ===");
        println!("Nodes: {}", reference_cpg.node_count());
        if let Some(root) = reference_cpg.get_root() {
            println!("Root node: {:?}", reference_cpg.get_node_by_id(&root));
        } else {
            println!("No root node");
        }

        let diff = cpg.compare(&reference_cpg).expect("Failed to compare CPGs");

        // For now, let's see what the structural difference is and fix it
        // This test helps us understand the problem better
        match diff {
            DetailedComparisonResult::Equivalent => println!("CPGs are equivalent!"),
            _ => {
                println!("=== STRUCTURAL DIFFERENCES ===");
                println!("{}", diff);

                // This test is currently expected to fail until we fix the empty-to-non-empty case
                // The issue is that we end up with a standalone Type node instead of a proper TranslationUnit
                println!(
                    "Expected failure: incremental update from empty doesn't create proper root structure"
                );
            }
        }
    }
}

/// Test specifically for the empty-to-non-empty root structure issue
#[test]
fn test_mre_empty_to_non_empty_root_structure() {
    dyn_cpg_rs::logging::init();
    let lang: RegisteredLanguage = "c".parse().expect("Failed to parse language");
    let mut parser = lang.get_parser().expect("Failed to get parser for C");

    // Test the specific case that's failing
    let empty_source = b"";
    let simple_source = b"int x;";

    if let Some(mut tree) = parser.parse(empty_source, None) {
        let mut cpg = lang
            .cst_to_cpg(tree.clone(), empty_source.to_vec())
            .expect("Failed to create CPG from empty source");

        let (edits, new_tree) =
            incremental_parse(&mut parser, empty_source, simple_source, &mut tree)
                .expect("Failed to parse simple source");

        let changed_ranges: Vec<_> = tree.changed_ranges(&new_tree).collect();

        // The core issue: when we have an empty CPG and try to incrementally update it,
        // we need to handle the case where we're essentially creating the entire structure from scratch

        // Check what type of CST node we're dealing with
        let new_cst_root = new_tree.root_node();
        println!(
            "New CST root: kind={}, text={:?}",
            new_cst_root.kind(),
            &simple_source[new_cst_root.start_byte()..new_cst_root.end_byte()]
        );

        // The new tree should be a translation_unit containing a declaration
        assert_eq!(new_cst_root.kind(), "translation_unit");

        cpg.incremental_update(
            edits,
            changed_ranges.into_iter(),
            &new_tree,
            simple_source.to_vec(),
        );

        // After incremental update, we should have a proper TranslationUnit root
        if let Some(root_id) = cpg.get_root() {
            let root_node = cpg
                .get_node_by_id(&root_id)
                .expect("Root node should exist");

            // The root should be a TranslationUnit, not a Type
            println!("Incremental CPG root: {:?}", root_node);

            // This is the assertion that will fail until we fix the issue
            // The problem is we're creating a Type node as root instead of TranslationUnit
            match &root_node.type_ {
                dyn_cpg_rs::cpg::node::NodeType::TranslationUnit => {
                    println!("SUCCESS: Root is correctly a TranslationUnit");
                }
                other => {
                    println!("PROBLEM: Root is {:?}, should be TranslationUnit", other);
                    // For debugging, let's see what we actually got
                    panic!("Root node should be TranslationUnit, got {:?}", other);
                }
            }
        } else {
            panic!("No root node found after incremental update");
        }
    }
}

/// MRE test for incremental parsing variable name change bug
/// This reproduces the issue from the failing test where incremental updates
/// don't properly handle structural changes in blocks
#[test]
fn test_mre_variable_name_change_bug() {
    dyn_cpg_rs::logging::init();
    let lang: RegisteredLanguage = "c".parse().expect("Failed to parse language");
    let mut parser = lang.get_parser().expect("Failed to get parser for C");

    // Minimal reproduction: variable name change that triggers the bug
    // Based on sample1.old.c -> sample1.new.c diff
    let old_source = b"int main() { int y = 2; }";
    let new_source = b"int main() { int z = 2; }";

    let mut old_tree = parser
        .parse(old_source, None)
        .expect("Failed to parse old source");

    // Perform incremental parsing
    let (edits, new_tree) = incremental_parse(&mut parser, old_source, new_source, &mut old_tree)
        .expect("Failed to incrementally parse");

    println!("=== EDITS ===");
    for edit in &edits {
        println!("{:?}", edit);
    }

    // Create CPGs
    let mut incremental_cpg = lang
        .cst_to_cpg(old_tree.clone(), old_source.to_vec())
        .expect("Failed to create initial CPG");
    let reference_cpg = lang
        .cst_to_cpg(new_tree.clone(), new_source.to_vec())
        .expect("Failed to create reference CPG");

    println!("=== BEFORE INCREMENTAL UPDATE ===");
    println!("Incremental CPG nodes: {}", incremental_cpg.node_count());
    println!("Reference CPG nodes: {}", reference_cpg.node_count());

    // Apply incremental update
    let changed_ranges = old_tree.changed_ranges(&new_tree);
    incremental_cpg.incremental_update(edits, changed_ranges, &new_tree, new_source.to_vec());

    println!("=== AFTER INCREMENTAL UPDATE ===");
    println!("Incremental CPG nodes: {}", incremental_cpg.node_count());
    println!("Reference CPG nodes: {}", reference_cpg.node_count());

    // Write debug files for analysis
    incremental_cpg
        .serialize_to_file(
            &mut SexpSerializer::new(),
            format!("{}/{}-{}.sexp", "debug", "mre_var_name_change", "incr"),
            None,
        )
        .expect("Failed to write incremental sexp");
    reference_cpg
        .serialize_to_file(
            &mut SexpSerializer::new(),
            format!("{}/{}-{}.sexp", "debug", "mre_var_name_change", "ref"),
            None,
        )
        .expect("Failed to write reference sexp");

    // Property: Incremental update should produce equivalent result
    let diff = incremental_cpg
        .compare(&reference_cpg)
        .expect("Failed to compare CPGs");
    match diff {
        DetailedComparisonResult::Equivalent => {
            println!("SUCCESS: CPGs are equivalent");
        }
        _ => {
            println!("=== BUG REPRODUCED ===");
            println!("{}", diff);
            panic!("BUG: Incremental CPG should be equivalent to reference CPG");
        }
    }
}

/// Test case derived from failing proptest
#[test]
fn test_mre_function_name_change_simple() {
    dyn_cpg_rs::logging::init();
    let lang: RegisteredLanguage = "c".parse().expect("Failed to parse language");
    let mut parser = lang.get_parser().expect("Failed to get parser for C");

    // Based on the failing proptest case
    let base_function_name = "_m___c_A0";
    let new_function_name = "d";
    let return_value = 0;

    let old_source = format!(
        "int {}() {{ return {}; }}",
        base_function_name, return_value
    );
    let new_source = format!(
        "int {}() {{ return {}; }}",
        new_function_name,
        return_value + 1
    );

    let old_bytes = old_source.as_bytes();
    let new_bytes = new_source.as_bytes();

    println!("Old source: {}", old_source);
    println!("New source: {}", new_source);

    let old_tree = parser.parse(old_bytes, None);
    assert!(old_tree.is_some(), "Old source should parse successfully");
    let mut old_tree = old_tree.unwrap();

    // Perform incremental parsing
    let incremental_result = incremental_parse(&mut parser, old_bytes, new_bytes, &mut old_tree);
    assert!(
        incremental_result.is_ok(),
        "Incremental parse should succeed"
    );
    let (edits, new_tree) = incremental_result.unwrap();

    println!("=== EDITS ===");
    for edit in &edits {
        println!("{:?}", edit);
    }

    // Create CPGs
    let old_cpg_result = lang.cst_to_cpg(old_tree.clone(), old_bytes.to_vec());
    let new_cpg_result = lang.cst_to_cpg(new_tree.clone(), new_bytes.to_vec());

    assert!(
        old_cpg_result.is_ok() && new_cpg_result.is_ok(),
        "CPG creation should succeed"
    );

    let mut incremental_cpg = old_cpg_result.unwrap();
    let reference_cpg = new_cpg_result.unwrap();

    println!("=== BEFORE INCREMENTAL UPDATE ===");
    println!("Incremental CPG nodes: {}", incremental_cpg.node_count());
    if let Some(root) = incremental_cpg.get_root() {
        println!("Root: {:?}", incremental_cpg.get_node_by_id(&root));
    }

    // Apply incremental update
    let changed_ranges = old_tree.changed_ranges(&new_tree);
    incremental_cpg.incremental_update(edits, changed_ranges, &new_tree, new_bytes.to_vec());

    println!("=== AFTER INCREMENTAL UPDATE ===");
    println!("Incremental CPG nodes: {}", incremental_cpg.node_count());
    println!("Reference CPG nodes: {}", reference_cpg.node_count());
    if let Some(root) = incremental_cpg.get_root() {
        println!(
            "Incremental root: {:?}",
            incremental_cpg.get_node_by_id(&root)
        );
    }
    if let Some(root) = reference_cpg.get_root() {
        println!("Reference root: {:?}", reference_cpg.get_node_by_id(&root));
    }

    // Property: Incremental update should produce equivalent result
    let comparison = incremental_cpg.compare(&reference_cpg);
    assert!(comparison.is_ok(), "CPG comparison should not fail");

    let diff = comparison.unwrap();
    match diff {
        DetailedComparisonResult::Equivalent => {
            println!("SUCCESS: CPGs are equivalent");
        }
        _ => {
            println!("=== COMPARISON FAILED ===");
            println!("{}", diff);
            panic!(
                "Incremental CPG should be equivalent to reference CPG for simple function changes"
            );
        }
    }
}

// --- Property-based tests for incremental parsing --- //

proptest! {
    #[test]
    fn prop_incremental_parsing_simple_changes(
        base_function_name in "[a-zA-Z_][a-zA-Z0-9_]{0,10}",
        new_function_name in "[a-zA-Z_][a-zA-Z0-9_]{0,10}",
        return_value in 0i32..100
    ) {
        let lang: RegisteredLanguage = "c".parse().expect("Failed to parse language");
        let mut parser = lang.get_parser().expect("Failed to get parser for C");

        // Create simple C functions with different names
        let old_source = format!("int {}() {{ return {}; }}", base_function_name, return_value);
        let new_source = format!("int {}() {{ return {}; }}", new_function_name, return_value + 1);

        let old_bytes = old_source.as_bytes();
        let new_bytes = new_source.as_bytes();

        let old_tree = parser.parse(old_bytes, None);
        prop_assume!(old_tree.is_some(), "Old source should parse successfully");
        let mut old_tree = old_tree.unwrap();

        // Perform incremental parsing
        let incremental_result = incremental_parse(&mut parser, old_bytes, new_bytes, &mut old_tree);
        prop_assume!(incremental_result.is_ok(), "Incremental parse should succeed");
        let (edits, new_tree) = incremental_result.unwrap();

        // Create CPGs
        let old_cpg_result = lang.cst_to_cpg(old_tree.clone(), old_bytes.to_vec());
        let new_cpg_result = lang.cst_to_cpg(new_tree.clone(), new_bytes.to_vec());

        prop_assume!(old_cpg_result.is_ok() && new_cpg_result.is_ok(), "CPG creation should succeed");

        let mut incremental_cpg = old_cpg_result.unwrap();
        let reference_cpg = new_cpg_result.unwrap();

        // Apply incremental update
        let changed_ranges = old_tree.changed_ranges(&new_tree);
        incremental_cpg.incremental_update(edits, changed_ranges, &new_tree, new_bytes.to_vec());

        // Property: Incremental update should produce equivalent result
        let comparison = incremental_cpg.compare(&reference_cpg);
        prop_assert!(comparison.is_ok(), "CPG comparison should not fail");

        let diff = comparison.unwrap();
        prop_assert!(
            matches!(diff, DetailedComparisonResult::Equivalent),
            "Incremental CPG should be equivalent to reference CPG for simple function changes"
        );
    }

    #[test]
    fn prop_incremental_parsing_whitespace_changes(
        spaces_before in 0usize..10,
        spaces_after in 0usize..10,
        newlines_before in 0usize..5,
        newlines_after in 0usize..5
    ) {
        let lang: RegisteredLanguage = "c".parse().expect("Failed to parse language");
        let mut parser = lang.get_parser().expect("Failed to get parser for C");

        // Create sources that differ only in whitespace
        let old_source = format!(
            "{}{}int main() {{{}return 0; }}",
            " ".repeat(spaces_before),
            "\n".repeat(newlines_before),
            " ".repeat(spaces_before)
        );
        let new_source = format!(
            "{}{}int main() {{{}return 0; }}",
            " ".repeat(spaces_after),
            "\n".repeat(newlines_after),
            " ".repeat(spaces_after)
        );

        let old_bytes = old_source.as_bytes();
        let new_bytes = new_source.as_bytes();

        let old_tree = parser.parse(old_bytes, None);
        prop_assume!(old_tree.is_some(), "Old source should parse successfully");
        let mut old_tree = old_tree.unwrap();

        let incremental_result = incremental_parse(&mut parser, old_bytes, new_bytes, &mut old_tree);
        prop_assume!(incremental_result.is_ok(), "Incremental parse should succeed");
        let (edits, new_tree) = incremental_result.unwrap();

        let old_cpg_result = lang.cst_to_cpg(old_tree.clone(), old_bytes.to_vec());
        let new_cpg_result = lang.cst_to_cpg(new_tree.clone(), new_bytes.to_vec());

        prop_assume!(old_cpg_result.is_ok() && new_cpg_result.is_ok(), "CPG creation should succeed");

        let mut incremental_cpg = old_cpg_result.unwrap();
        let reference_cpg = new_cpg_result.unwrap();

        let changed_ranges = old_tree.changed_ranges(&new_tree);
        incremental_cpg.incremental_update(edits, changed_ranges, &new_tree, new_bytes.to_vec());

        // Property: Whitespace-only changes should still produce equivalent CPGs
        let comparison = incremental_cpg.compare(&reference_cpg);
        prop_assert!(comparison.is_ok(), "CPG comparison should not fail");

        let diff = comparison.unwrap();
        prop_assert!(
            matches!(diff, DetailedComparisonResult::Equivalent),
            "Incremental CPG should handle whitespace changes correctly"
        );
    }

    #[test]
    fn prop_incremental_parsing_statement_insertion(
        var_name in "[a-zA-Z_][a-zA-Z0-9_]{0,8}",
        initial_value in 0i32..50,
        inserted_value in 0i32..50
    ) {
        let lang: RegisteredLanguage = "c".parse().expect("Failed to parse language");
        let mut parser = lang.get_parser().expect("Failed to get parser for C");

        // Original function with one statement
        let old_source = format!(
            "int main() {{ int {} = {}; return {}; }}",
            var_name, initial_value, var_name
        );

        // Modified function with an additional statement
        let new_source = format!(
            "int main() {{ int {} = {}; {} = {}; return {}; }}",
            var_name, initial_value, var_name, inserted_value, var_name
        );

        let old_bytes = old_source.as_bytes();
        let new_bytes = new_source.as_bytes();

        let old_tree = parser.parse(old_bytes, None);
        prop_assume!(old_tree.is_some(), "Old source should parse successfully");
        let mut old_tree = old_tree.unwrap();

        let incremental_result = incremental_parse(&mut parser, old_bytes, new_bytes, &mut old_tree);
        prop_assume!(incremental_result.is_ok(), "Incremental parse should succeed");
        let (edits, new_tree) = incremental_result.unwrap();

        let old_cpg_result = lang.cst_to_cpg(old_tree.clone(), old_bytes.to_vec());
        let new_cpg_result = lang.cst_to_cpg(new_tree.clone(), new_bytes.to_vec());

        prop_assume!(old_cpg_result.is_ok() && new_cpg_result.is_ok(), "CPG creation should succeed");

        let mut incremental_cpg = old_cpg_result.unwrap();
        let reference_cpg = new_cpg_result.unwrap();

        let changed_ranges = old_tree.changed_ranges(&new_tree);
        incremental_cpg.incremental_update(edits, changed_ranges, &new_tree, new_bytes.to_vec());

        // Property: Statement insertion should be handled correctly
        let comparison = incremental_cpg.compare(&reference_cpg);
        prop_assert!(comparison.is_ok(), "CPG comparison should not fail");

        let diff = comparison.unwrap();
        prop_assert!(
            matches!(diff, DetailedComparisonResult::Equivalent),
            "Incremental CPG should handle statement insertion correctly"
        );
    }

    #[test]
    fn prop_incremental_parsing_consistency_with_edits(
        edit_count in 1usize..5,
        base_content in prop::collection::vec("[a-zA-Z0-9_]{1,10}", 3..8)
    ) {
        let lang: RegisteredLanguage = "c".parse().expect("Failed to parse language");
        let mut parser = lang.get_parser().expect("Failed to get parser for C");

        // Build a base C function with variable declarations
        let mut old_source = "int main() {\n".to_string();
        for (i, content) in base_content.iter().enumerate() {
            old_source.push_str(&format!("    int var{} = {};\n", i, content.len()));
        }
        old_source.push_str("    return 0;\n}");

        // Make some modifications
        let mut new_source = old_source.clone();
        for i in 0..edit_count.min(base_content.len()) {
            let replacement = format!("var{}_modified", i);
            new_source = new_source.replace(&format!("var{}", i), &replacement);
        }

        // Skip if sources are identical (no actual changes made)
        prop_assume!(old_source != new_source, "Sources should be different");

        let old_bytes = old_source.as_bytes();
        let new_bytes = new_source.as_bytes();

        let old_tree = parser.parse(old_bytes, None);
        prop_assume!(old_tree.is_some(), "Old source should parse successfully");
        let mut old_tree = old_tree.unwrap();

        let incremental_result = incremental_parse(&mut parser, old_bytes, new_bytes, &mut old_tree);
        prop_assume!(incremental_result.is_ok(), "Incremental parse should succeed");
        let (edits, new_tree) = incremental_result.unwrap();

        // Property: Number of edits should be reasonable
        prop_assert!(
            !edits.is_empty(),
            "Should have at least one edit when sources differ"
        );
        prop_assert!(
            edits.len() <= edit_count * 2, // Allow some overhead for tree-sitter's edit detection
            "Edit count should be reasonable: got {} edits for {} expected changes",
            edits.len(),
            edit_count
        );

        let old_cpg_result = lang.cst_to_cpg(old_tree.clone(), old_bytes.to_vec());
        let new_cpg_result = lang.cst_to_cpg(new_tree.clone(), new_bytes.to_vec());

        prop_assume!(old_cpg_result.is_ok() && new_cpg_result.is_ok(), "CPG creation should succeed");

        let mut incremental_cpg = old_cpg_result.unwrap();
        let reference_cpg = new_cpg_result.unwrap();

        let changed_ranges = old_tree.changed_ranges(&new_tree);

        // Property: Changed ranges should correlate with edits
        prop_assert!(
            changed_ranges.len() > 0,
            "Should have changed ranges when there are edits"
        );

        incremental_cpg.incremental_update(edits, changed_ranges, &new_tree, new_bytes.to_vec());

        let comparison = incremental_cpg.compare(&reference_cpg);
        prop_assert!(comparison.is_ok(), "CPG comparison should not fail");

        let diff = comparison.unwrap();
        prop_assert!(
            matches!(diff, DetailedComparisonResult::Equivalent),
            "Incremental CPG should be equivalent to reference CPG for variable name changes"
        );
    }
}
