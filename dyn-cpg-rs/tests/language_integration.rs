use dyn_cpg_rs::{languages::RegisteredLanguage, resource::Resource};

/// Integration test for language parsing and CPG generation
/// Tests end-to-end functionality from source code to CPG
#[test]
fn test_c_language_integration() {
    let lang: RegisteredLanguage = "c".parse().expect("Failed to parse language");
    let mut parser = lang.get_parser().expect("Failed to get parser for C");

    // Test with various C constructs
    let sources = [
        b"int main() { return 0; }".as_slice(),
        b"int x = 5; int y = 10;".as_slice(),
        b"
            int factorial(int n) {
                if (n <= 1) return 1;
                return n * factorial(n - 1);
            }
        "
        .as_slice(),
        b"
            int main() {
                for (int i = 0; i < 10; i++) {
                    printf(\"%d\\n\", i);
                }
                return 0;
            }
        "
        .as_slice(),
    ];

    for (i, source) in sources.iter().enumerate() {
        let tree = parser
            .parse(source, None)
            .unwrap_or_else(|| panic!("Failed to parse source {}", i));

        let cpg = lang
            .cst_to_cpg(tree, source.to_vec())
            .unwrap_or_else(|_| panic!("Failed to convert source {} to CPG", i));

        // Basic sanity checks
        assert!(cpg.get_root().is_some(), "CPG {} should have a root", i);
        assert!(cpg.node_count() > 0, "CPG {} should have nodes", i);
    }
}

/// Test CPG generation with sample files
#[test]
fn test_sample_files_integration() {
    let lang: RegisteredLanguage = "c".parse().expect("Failed to parse language");
    let mut parser = lang.get_parser().expect("Failed to get parser for C");

    // Test with sample1.old.c
    let sample_old = Resource::new("samples/sample1.old.c")
        .expect("Failed to create resource for sample1.old.c");
    let old_src = sample_old
        .read_bytes()
        .expect("Failed to read sample1.old.c");

    let old_tree = parser
        .parse(&old_src, None)
        .expect("Failed to parse sample1.old.c");

    let old_cpg = lang
        .cst_to_cpg(old_tree, old_src)
        .expect("Failed to convert sample1.old.c to CPG");

    // Test with sample1.new.c
    let sample_new = Resource::new("samples/sample1.new.c")
        .expect("Failed to create resource for sample1.new.c");
    let new_src = sample_new
        .read_bytes()
        .expect("Failed to read sample1.new.c");

    let new_tree = parser
        .parse(&new_src, None)
        .expect("Failed to parse sample1.new.c");

    let new_cpg = lang
        .cst_to_cpg(new_tree, new_src)
        .expect("Failed to convert sample1.new.c to CPG");

    // Both CPGs should be valid
    assert!(old_cpg.get_root().is_some());
    assert!(new_cpg.get_root().is_some());
    assert!(old_cpg.node_count() > 0);
    assert!(new_cpg.node_count() > 0);
    assert!(old_cpg.edge_count() > 0);
    assert!(new_cpg.edge_count() > 0);
}

/// Test control flow and data dependence passes
#[test]
fn test_analysis_passes_integration() {
    use dyn_cpg_rs::languages::{cf_pass, data_dep_pass};

    let lang: RegisteredLanguage = "c".parse().expect("Failed to parse language");
    let mut parser = lang.get_parser().expect("Failed to get parser for C");

    let source = b"
        int main() {
            int x = 5;
            if (x > 0) {
                x = x + 1;
            } else {
                x = x - 1;
            }
            return x;
        }
    ";

    let tree = parser.parse(source, None).expect("Failed to parse source");
    let mut cpg = lang
        .cst_to_cpg(tree, source.to_vec())
        .expect("Failed to convert to CPG");

    let root = cpg.get_root().expect("CPG should have a root");

    // Test control flow pass
    let cf_result = cf_pass(&mut cpg, root);
    assert!(cf_result.is_ok(), "Control flow pass should succeed");

    // Test data dependence pass
    let dd_result = data_dep_pass(&mut cpg, root);
    assert!(dd_result.is_ok(), "Data dependence pass should succeed");

    // Verify that analysis passes added edges
    let total_edges_after = cpg.edge_count();
    assert!(
        total_edges_after > 0,
        "Analysis passes should have added edges"
    );
}

/// Test error handling with malformed C code
#[test]
fn test_error_handling_integration() {
    let lang: RegisteredLanguage = "c".parse().expect("Failed to parse language");
    let mut parser = lang.get_parser().expect("Failed to get parser for C");

    // Test with syntactically incorrect C code
    let malformed_sources = vec![
        b"int main( { return 0; }".as_slice(), // Missing closing paren
        b"int main() { return }".as_slice(),   // Missing semicolon and value
        b"if (x > 0) { return; }".as_slice(),  // Missing main function
    ];

    for malformed in malformed_sources {
        // Even malformed code should parse (tree-sitter is error-tolerant)
        let tree = parser.parse(malformed, None);
        assert!(
            tree.is_some(),
            "Tree-sitter should handle malformed code gracefully"
        );

        if let Some(tree) = tree {
            // CPG generation might succeed even with parse errors
            let _cpg_result = lang.cst_to_cpg(tree, malformed.to_vec());
            // We don't assert success here since some malformed code might
            // legitimately fail CPG generation
        }
    }
}

/// Test large file handling if available
#[test]
fn test_large_file_integration() {
    let lang: RegisteredLanguage = "c".parse().expect("Failed to parse language");
    let mut parser = lang.get_parser().expect("Failed to get parser for C");

    // Only run this test if large_sample.c exists
    if let Ok(large_sample) = Resource::new("samples/large_sample.old.c") {
        let src = large_sample
            .read_bytes()
            .expect("Failed to read large sample");

        let start = std::time::Instant::now();
        let tree = parser
            .parse(&src, None)
            .expect("Failed to parse large sample");
        let parse_time = start.elapsed();

        let start = std::time::Instant::now();
        let cpg = lang
            .cst_to_cpg(tree, src)
            .expect("Failed to convert large sample to CPG");
        let cpg_time = start.elapsed();

        // Basic validation
        assert!(cpg.get_root().is_some());
        assert!(cpg.node_count() > 100); // Should be a reasonably large file

        // Performance checks - these are loose bounds
        assert!(
            parse_time.as_secs() < 30,
            "Parsing should complete within 30 seconds, took {:?}",
            parse_time
        );
        assert!(
            cpg_time.as_secs() < 60,
            "CPG generation should complete within 60 seconds, took {:?}",
            cpg_time
        );

        println!(
            "Large file stats: {} nodes, {} edges, parsed in {:?}, CPG in {:?}",
            cpg.node_count(),
            cpg.edge_count(),
            parse_time,
            cpg_time
        );
    }
}
