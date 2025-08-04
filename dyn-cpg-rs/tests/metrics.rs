use std::path::PathBuf;

use dyn_cpg_rs::{languages::RegisteredLanguage, resource::Resource};
use glob::glob;

fn parse_glob_with_git(
    pattern: &str,
    repo_root: Option<&PathBuf>,
    revision: Option<&str>,
) -> Vec<Result<Resource, String>> {
    let matches: Vec<_> = glob(pattern)
        .expect("Failed to read glob pattern")
        .map(|path_result| {
            path_result
                .map_err(|e| format!("Glob error: {}", e))
                .and_then(|path| {
                    let mut resource = Resource::new(&path).map_err(|e| {
                        format!("Failed to create resource for '{:?}': {}", path, e)
                    })?;

                    if let (Some(repo), Some(rev)) = (repo_root, revision) {
                        let repo_path: PathBuf = repo.clone();
                        resource = resource.with_git(rev.to_string(), repo_path).map_err(|e| {
                            format!("Failed to set Git context for '{:?}': {}", path, e)
                        })?;
                    }

                    Ok(resource)
                })
        })
        .collect();

    matches
}

/*
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
     */

#[test]
fn test_incr_perf_gv() {
    // Test the incremental parsing performance on the Graphviz repo
    let depth: u8 = 5;
    let repo_root: PathBuf = "./repos/graphviz".into();
    let repo_root = repo_root
        .canonicalize()
        .expect("Failed to canonicalize repo root");

    let lang: RegisteredLanguage = "c".parse().expect("Failed to parse language");
    let mut parser = lang.get_parser().expect("Failed to get parser for C");

    let pattern = format!("{}/**/*.c", repo_root.display());

    // Go back "depth" commits and parse all of the C files in "repo/graphviz"
    // Then step forward one commit at a time, parsing the changed files (incrementally and fully)
    // Log all of the benchmarks

    let resources = parse_glob_with_git(
        &pattern,
        Some(&repo_root),
        Some(format!("HEAD~{}", depth).as_str()),
    );
    let mut resources: Vec<_> = resources.into_iter().filter_map(Result::ok).collect();
    resources.sort_by_key(|r| r.raw_path().to_string_lossy().to_string());
    println!(
        "Found {} C files in repo: {}",
        resources.len(),
        repo_root.display()
    );
    let cpgs = resources
        .iter()
        .enumerate()
        .map(|(idx, r)| {
            let src = r.read_bytes().expect("Failed to read resource bytes");
            let tree = parser
                .parse(src.clone(), None)
                .expect("Failed to parse source");
            lang.cst_to_cpg(tree, src).expect(
                format!(
                    "Failed to convert CST to CPG for file {:?} at index {}",
                    r.raw_path(),
                    idx
                )
                .as_str(),
            )
        })
        .collect::<Vec<_>>();
}
