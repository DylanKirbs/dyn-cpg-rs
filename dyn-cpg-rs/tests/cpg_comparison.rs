use dyn_cpg_rs::{
    cpg::{DetailedComparisonResult, FunctionComparisonResult},
    languages::RegisteredLanguage,
    resource::Resource,
};

/// Integration test for CPG comparison functionality
/// Tests semantic comparison of CPGs generated from real C code
#[test]
fn test_cpg_comparison_with_real_code() {
    let lang: RegisteredLanguage = "c".parse().expect("Failed to parse language");
    let mut parser = lang.get_parser().expect("Failed to get parser for C");

    // Test with sample files
    let s1 = Resource::new("samples/sample1.old.c")
        .expect("Failed to create resource for sample1.old.c");
    let s2 = Resource::new("samples/sample1.new.c")
        .expect("Failed to create resource for sample1.new.c");

    let src1 = s1.read_bytes().expect("Failed to read sample1.old.c");
    let src2 = s2.read_bytes().expect("Failed to read sample1.new.c");

    // Parse both files
    let tree1 = parser
        .parse(&src1, None)
        .expect("Failed to parse sample1.old.c");
    let tree2 = parser
        .parse(&src2, None)
        .expect("Failed to parse sample1.new.c");

    // Generate CPGs
    let cpg1 = lang
        .cst_to_cpg(tree1, src1)
        .expect("Failed to convert old tree to CPG");
    let cpg2 = lang
        .cst_to_cpg(tree2, src2)
        .expect("Failed to convert new tree to CPG");

    // Compare CPGs
    let result = cpg1.compare(&cpg2).expect("Failed to compare CPGs");

    // The comparison should detect differences (since old vs new are different)
    match result {
        DetailedComparisonResult::Equivalent => {
            panic!("Expected differences between old and new sample files");
        }
        DetailedComparisonResult::StructuralMismatch { .. } => {
            // This is expected - old and new samples should differ
        }
    }
}

/// Test CPG comparison with identical content
#[test]
fn test_cpg_comparison_identical() {
    let lang: RegisteredLanguage = "c".parse().expect("Failed to parse language");
    let mut parser = lang.get_parser().expect("Failed to get parser for C");

    let source = b"int main() { return 0; }";

    // Parse the same source twice
    let tree1 = parser
        .parse(source, None)
        .expect("Failed to parse source 1");
    let tree2 = parser
        .parse(source, None)
        .expect("Failed to parse source 2");

    // Generate CPGs from identical sources
    let cpg1 = lang
        .cst_to_cpg(tree1, source.to_vec())
        .expect("Failed to convert tree1 to CPG");
    let cpg2 = lang
        .cst_to_cpg(tree2, source.to_vec())
        .expect("Failed to convert tree2 to CPG");

    // Compare CPGs
    let result = cpg1.compare(&cpg2).expect("Failed to compare CPGs");
    assert!(
        matches!(result, DetailedComparisonResult::Equivalent),
        "Identical sources should produce equivalent CPGs"
    );
}

/// Test CPG comparison with different function names
#[test]
fn test_cpg_comparison_different_functions() {
    let lang: RegisteredLanguage = "c".parse().expect("Failed to parse language");
    let mut parser = lang.get_parser().expect("Failed to get parser for C");

    let source1 = b"int main() { return 0; }";
    let source2 = b"int test() { return 0; }";

    let tree1 = parser
        .parse(source1, None)
        .expect("Failed to parse source 1");
    let tree2 = parser
        .parse(source2, None)
        .expect("Failed to parse source 2");

    let cpg1 = lang
        .cst_to_cpg(tree1, source1.to_vec())
        .expect("Failed to convert tree1 to CPG");
    let cpg2 = lang
        .cst_to_cpg(tree2, source2.to_vec())
        .expect("Failed to convert tree2 to CPG");

    let result = cpg1.compare(&cpg2).expect("Failed to compare CPGs");
    match result {
        DetailedComparisonResult::StructuralMismatch {
            only_in_left,
            only_in_right,
            ..
        } => {
            assert!(only_in_left.contains(&"main".to_string()));
            assert!(only_in_right.contains(&"test".to_string()));
        }
        _ => panic!("Expected structural mismatch for different function names"),
    }
}

/// Test CPG comparison with complex structural differences
#[test]
fn test_cpg_comparison_complex_differences() {
    let lang: RegisteredLanguage = "c".parse().expect("Failed to parse language");
    let mut parser = lang.get_parser().expect("Failed to get parser for C");

    let source1 = b"
        int main() { 
            int x = 5;
            return x; 
        }
    ";
    let source2 = b"
        int main() { 
            int x = 5;
            int y = 10;
            return x + y; 
        }
        int helper() {
            return 42;
        }
    ";

    let tree1 = parser
        .parse(source1, None)
        .expect("Failed to parse source 1");
    let tree2 = parser
        .parse(source2, None)
        .expect("Failed to parse source 2");

    let cpg1 = lang
        .cst_to_cpg(tree1, source1.to_vec())
        .expect("Failed to convert tree1 to CPG");
    let cpg2 = lang
        .cst_to_cpg(tree2, source2.to_vec())
        .expect("Failed to convert tree2 to CPG");

    let result = cpg1.compare(&cpg2).expect("Failed to compare CPGs");
    match result {
        DetailedComparisonResult::StructuralMismatch {
            only_in_right,
            function_mismatches,
            ..
        } => {
            // Should detect new helper function in right CPG
            assert!(only_in_right.contains(&"helper".to_string()));

            // Should detect changes in main function
            let main_mismatch = function_mismatches.iter().find(|m| match m {
                FunctionComparisonResult::Mismatch { function_name, .. } => function_name == "main",
                _ => false,
            });
            assert!(
                main_mismatch.is_some(),
                "Should detect main function changes"
            );
        }
        _ => panic!("Expected structural mismatch for complex differences"),
    }
}

/// Test performance of CPG comparison on larger files
#[test]
fn test_cpg_comparison_performance() {
    let lang: RegisteredLanguage = "c".parse().expect("Failed to parse language");
    let mut parser = lang.get_parser().expect("Failed to get parser for C");

    // Check if large sample exists
    if let Ok(large_sample) = Resource::new("samples/large_sample.old.c") {
        let src = large_sample
            .read_bytes()
            .expect("Failed to read large sample");

        let tree1 = parser
            .parse(&src, None)
            .expect("Failed to parse large sample 1");
        let tree2 = parser
            .parse(&src, None)
            .expect("Failed to parse large sample 2");

        let cpg1 = lang
            .cst_to_cpg(tree1, src.clone())
            .expect("Failed to convert large tree1 to CPG");
        let cpg2 = lang
            .cst_to_cpg(tree2, src)
            .expect("Failed to convert large tree2 to CPG");

        let start = std::time::Instant::now();
        let result = cpg1.compare(&cpg2).expect("Failed to compare large CPGs");
        let duration = start.elapsed();

        assert!(
            matches!(result, DetailedComparisonResult::Equivalent),
            "Identical large sources should produce equivalent CPGs"
        );

        // Performance check - should complete within reasonable time
        assert!(
            duration.as_secs() < 10,
            "CPG comparison should complete within 10 seconds, took {:?}",
            duration
        );
    }
}

/// Debug test to inspect the Tree-sitter parse tree structure
#[test]
fn debug_tree_sitter_parsing() {
    let lang: RegisteredLanguage = "c".parse().expect("Failed to parse language");
    let mut parser = lang.get_parser().expect("Failed to get parser for C");

    let source1 = b"int main() { return 0; }";
    let source2 = b"int test() { return 0; }";

    let tree1 = parser
        .parse(source1, None)
        .expect("Failed to parse source 1");
    let tree2 = parser
        .parse(source2, None)
        .expect("Failed to parse source 2");

    println!("Tree 1 S-expression: {}", tree1.root_node().to_sexp());
    println!("Tree 2 S-expression: {}", tree2.root_node().to_sexp());

    let cpg1 = lang
        .cst_to_cpg(tree1, source1.to_vec())
        .expect("Failed to convert tree1 to CPG");
    let cpg2 = lang
        .cst_to_cpg(tree2, source2.to_vec())
        .expect("Failed to convert tree2 to CPG");

    println!("CPG1 nodes: {}", cpg1.node_count());
    println!("CPG2 nodes: {}", cpg2.node_count());

    // Now we can call the public method
    if let Some(root1) = cpg1.get_root() {
        let functions1 = cpg1.get_top_level_functions(root1).unwrap();
        println!("CPG1 functions: {:?}", functions1);
    }

    if let Some(root2) = cpg2.get_root() {
        let functions2 = cpg2.get_top_level_functions(root2).unwrap();
        println!("CPG2 functions: {:?}", functions2);
    }

    let result = cpg1.compare(&cpg2).expect("Failed to compare CPGs");
    println!("Comparison result: {}", result);
}
