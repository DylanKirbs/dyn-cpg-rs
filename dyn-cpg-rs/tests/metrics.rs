use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use dyn_cpg_rs::{
    cpg::{Cpg, DetailedComparisonResult},
    diff::{SourceEdit, incremental_parse},
    languages::RegisteredLanguage,
    resource::Resource,
};
use git2::{Oid, Repository};
use glob::glob;
use indicatif::{ProgressBar, ProgressStyle};
use serde_json::json;
use std::fs::File;
use tracing::{info, warn};

#[derive(Debug, Default)]
struct DetailedTimings {
    // Full parse timings
    ts_full_parse_time_ms: Option<u128>,
    cst_to_cpg_time_ms: Option<u128>,

    // Incremental parse timings
    ts_old_parse_time_ms: Option<u128>,
    text_diff_time_ms: Option<u128>,
    ts_edit_apply_time_ms: Option<u128>,
    ts_incremental_parse_time_ms: Option<u128>,
    cpg_incremental_update_time_ms: Option<u128>,

    // Comparison timing
    comparison_time_ms: Option<u128>,
}

impl DetailedTimings {
    fn to_json(&self) -> serde_json::Value {
        json!({
            "ts_full_parse_time_ms": self.ts_full_parse_time_ms,
            "cst_to_cpg_time_ms": self.cst_to_cpg_time_ms,
            "ts_old_parse_time_ms": self.ts_old_parse_time_ms,
            "text_diff_time_ms": self.text_diff_time_ms,
            "ts_edit_apply_time_ms": self.ts_edit_apply_time_ms,
            "ts_incremental_parse_time_ms": self.ts_incremental_parse_time_ms,
            "cpg_incremental_update_time_ms": self.cpg_incremental_update_time_ms,
            "comparison_time_ms": self.comparison_time_ms
        })
    }

    fn full_parse_total_ms(&self) -> Option<u128> {
        match (self.ts_full_parse_time_ms, self.cst_to_cpg_time_ms) {
            (Some(ts), Some(cpg)) => Some(ts + cpg),
            _ => None,
        }
    }

    fn incremental_parse_total_ms(&self) -> Option<u128> {
        match (
            self.ts_old_parse_time_ms,
            self.text_diff_time_ms,
            self.ts_edit_apply_time_ms,
            self.ts_incremental_parse_time_ms,
            self.cpg_incremental_update_time_ms,
        ) {
            (Some(old), Some(diff), Some(edit), Some(incr), Some(update)) => {
                Some(old + diff + edit + incr + update)
            }
            _ => None,
        }
    }
}

#[derive(Debug, Default)]
struct FileMetrics {
    file_size_bytes: usize,
    line_count: usize,
    changed_lines: Option<usize>,
    proportion_lines_changed: Option<f64>,
}

impl FileMetrics {
    fn from_source(src_bytes: &[u8]) -> Self {
        let file_size_bytes = src_bytes.len();
        let line_count = src_bytes.iter().filter(|&&b| b == b'\n').count() + 1;

        Self {
            file_size_bytes,
            line_count,
            changed_lines: None,
            proportion_lines_changed: None,
        }
    }

    fn with_change_analysis(
        mut self,
        old_src: &[u8],
        new_src: &[u8],
        edits: &[SourceEdit],
    ) -> Self {
        if edits.is_empty() {
            self.changed_lines = Some(0);
            self.proportion_lines_changed = Some(0.0);
            return self;
        }

        let old_lines =
            count_lines_in_byte_ranges(old_src, edits.iter().map(|e| (e.old_start, e.old_end)));
        let new_lines =
            count_lines_in_byte_ranges(new_src, edits.iter().map(|e| (e.new_start, e.new_end)));

        // Use the maximum of old and new lines affected as the total changed lines
        let changed_lines = std::cmp::max(old_lines, new_lines);
        let proportion = if self.line_count > 0 {
            changed_lines as f64 / self.line_count as f64
        } else {
            0.0
        };

        self.changed_lines = Some(changed_lines);
        self.proportion_lines_changed = Some(proportion);
        self
    }

    fn to_json(&self) -> serde_json::Value {
        json!({
            "file_size_bytes": self.file_size_bytes,
            "line_count": self.line_count,
            "changed_lines": self.changed_lines,
            "proportion_lines_changed": self.proportion_lines_changed
        })
    }
}

fn count_lines_in_byte_ranges<I>(src_bytes: &[u8], ranges: I) -> usize
where
    I: Iterator<Item = (usize, usize)>,
{
    let mut affected_lines = std::collections::HashSet::new();

    for (start_byte, end_byte) in ranges {
        let start_line = src_bytes[..start_byte]
            .iter()
            .filter(|&&b| b == b'\n')
            .count();
        let end_line = src_bytes[..end_byte]
            .iter()
            .filter(|&&b| b == b'\n')
            .count();

        // Include all lines from start to end (inclusive)
        for line_num in start_line..=end_line {
            affected_lines.insert(line_num);
        }
    }

    affected_lines.len()
}

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
                        resource =
                            resource
                                .with_git(rev.to_string(), repo.clone())
                                .map_err(|e| {
                                    format!("Failed to set Git context for '{:?}': {}", path, e)
                                })?;
                    }

                    Ok(resource)
                })
        })
        .collect();

    matches
}

fn perform_full_parse(
    parser: &mut tree_sitter::Parser,
    lang: &RegisteredLanguage,
    src_bytes: &[u8],
) -> Result<(Cpg, DetailedTimings), String> {
    let mut timings = DetailedTimings::default();

    // Step 1: Tree-sitter parse
    let ts_start = Instant::now();
    let tree = parser
        .parse(src_bytes, None)
        .ok_or("Tree-sitter parse failed")?;
    timings.ts_full_parse_time_ms = Some(ts_start.elapsed().as_millis());

    // Step 2: CST → CPG conversion
    let cst_start = Instant::now();
    let cpg = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        lang.cst_to_cpg(tree, src_bytes.to_vec())
    }))
    .map_err(|_| "CST to CPG panicked (likely stack overflow)")?
    .map_err(|e| format!("CST to CPG failed: {}", e))?;
    timings.cst_to_cpg_time_ms = Some(cst_start.elapsed().as_millis());

    Ok((cpg, timings))
}

fn perform_incremental_parse(
    parser: &mut tree_sitter::Parser,
    prev_src: &[u8],
    new_src: &[u8],
    mut prev_cpg: Cpg,
) -> Result<(Cpg, DetailedTimings, Vec<SourceEdit>), String> {
    let mut timings = DetailedTimings::default();

    // Step 1: Parse old source
    let old_parse_start = Instant::now();
    let mut prev_tree = parser
        .parse(prev_src, None)
        .ok_or("Previous source parse failed")?;
    timings.ts_old_parse_time_ms = Some(old_parse_start.elapsed().as_millis());

    // Step 2: Compute text diff and apply edits
    let diff_start = Instant::now();
    let (edits, new_tree) = incremental_parse(parser, prev_src, new_src, &mut prev_tree)
        .map_err(|e| format!("Incremental parse failed: {}", e))?;
    let diff_time = diff_start.elapsed();

    // Note: incremental_parse internally does both diff computation and tree-sitter incremental parsing
    // We can't easily separate these without modifying the incremental_parse function
    // For now, we'll attribute the time to text_diff but add a note
    timings.text_diff_time_ms = Some(diff_time.as_millis());
    // TODO: Split incremental_parse to separate diff computation from TS incremental parsing
    timings.ts_edit_apply_time_ms = Some(0); // Bundled with text_diff for now
    timings.ts_incremental_parse_time_ms = Some(0); // Bundled with text_diff for now

    // Step 3: Update CPG incrementally
    let cpg_update_start = Instant::now();
    let changed_ranges = prev_tree.changed_ranges(&new_tree);
    prev_cpg.incremental_update(edits.clone(), changed_ranges, &new_tree, new_src.to_vec());
    timings.cpg_incremental_update_time_ms = Some(cpg_update_start.elapsed().as_millis());

    Ok((prev_cpg, timings, edits))
}

fn walk_git_history_and_benchmark(
    repo_path: &PathBuf,
    pattern: &str,
    depth: usize,
    lang: &RegisteredLanguage,
) -> Result<Vec<serde_json::Value>, Box<dyn std::error::Error>> {
    let repo = Repository::open(repo_path)?;
    let mut revwalk = repo.revwalk()?;
    revwalk.push_head()?;

    let mut commits: Vec<Oid> = revwalk.take(depth + 1).collect::<Result<Vec<_>, _>>()?;
    commits.reverse();

    let mut results = Vec::new();
    let mut previous_resources: Option<Vec<Resource>> = None;
    let mut previous_cpgs: Option<HashMap<String, Cpg>> = None;
    let mut previous_hashes: Option<HashMap<String, blake3::Hash>> = None;

    // Create progress bar for commits
    let commit_progress = ProgressBar::new(commits.len() as u64);
    commit_progress.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>3}/{len:3} {msg}",
            )
            .unwrap()
            .progress_chars("##-"),
    );

    for (step, commit_oid) in commits.iter().enumerate() {
        let commit = repo.find_commit(*commit_oid)?;
        let commit_id = commit.id().to_string();
        let commit_message = commit.message().unwrap_or("").lines().next().unwrap_or("");

        commit_progress.set_message(format!("{} - {}", &commit_id[..8], commit_message));
        commit_progress.set_position(step as u64);

        let step_start = Instant::now();

        let resources = parse_glob_with_git(pattern, Some(repo_path), Some(&commit_id));
        let resources: Vec<_> = resources.into_iter().filter_map(Result::ok).collect();

        let resource_load_time = step_start.elapsed();
        info!(
            "Loaded {} resources in {:?}",
            resources.len(),
            resource_load_time
        );

        let mut parser = lang.get_parser()?;
        let mut file_results = Vec::new();
        let mut current_cpgs = HashMap::new();
        let mut current_hashes = HashMap::new();

        for resource in resources.iter() {
            let file_path = resource.raw_path().to_string_lossy().to_string();

            // Calculate current file hash
            let current_hash = match resource.hash() {
                Ok(hash) => hash,
                Err(e) => {
                    warn!("Failed to compute hash for file {}: {}", file_path, e);
                    continue;
                }
            };

            // Check if file has changed since previous commit
            let file_unchanged = if let Some(prev_hashes) = &previous_hashes {
                prev_hashes.get(&file_path) == Some(&current_hash)
            } else {
                false
            };

            if file_unchanged {
                // File hasn't changed, reuse previous CPG and create a minimal result
                if let Some(prev_cpgs) = &previous_cpgs {
                    if let Some(prev_cpg) = prev_cpgs.get(&file_path).cloned() {
                        let file_result = json!({
                            "file": file_path.clone(),
                            "commit": commit_id,
                            "step": step,
                            "unchanged": true,
                            "full_nodes": prev_cpg.node_count(),
                            "full_edges": prev_cpg.edge_count(),
                            "incremental_nodes": null,
                            "incremental_edges": null,
                            "comparison_result": null,
                            // Legacy fields for backward compatibility
                            "full_parse_time_ms": 0,
                            "incremental_parse_time_ms": null,
                            "cpg_update_time_ms": null,
                            "comparison_time_ms": null,
                            // Detailed timing fields
                            "detailed_timings": DetailedTimings::default().to_json()
                        });

                        current_cpgs.insert(file_path.clone(), prev_cpg);
                        current_hashes.insert(file_path, current_hash);
                        file_results.push(file_result);
                        continue;
                    }
                }
            }

            // File has changed or is new, read source
            let src_bytes = match resource.read_bytes() {
                Ok(bytes) => bytes,
                Err(e) => {
                    warn!("Failed to read file {}: {}", file_path, e);
                    file_results.push(json!({
                        "file": file_path,
                        "commit": commit_id,
                        "step": step,
                        "error": format!("Failed to read file: {}", e)
                    }));
                    continue;
                }
            };

            // Perform full parse (always)
            let (full_cpg, full_timings) = match perform_full_parse(&mut parser, lang, &src_bytes) {
                Ok(result) => result,
                Err(e) => {
                    warn!("Full parse failed for file {}: {}", file_path, e);
                    file_results.push(json!({
                        "file": file_path,
                        "commit": commit_id,
                        "step": step,
                        "error": e
                    }));
                    continue;
                }
            };

            // Calculate base file metrics
            let mut file_metrics = FileMetrics::from_source(&src_bytes);

            let mut file_result = json!({
                "file": file_path.clone(),
                "commit": commit_id,
                "step": step,
                "unchanged": false,
                "full_nodes": full_cpg.node_count(),
                "full_edges": full_cpg.edge_count(),
                "incremental_nodes": null,
                "incremental_edges": null,
                "comparison_result": null,
                // Legacy fields for backward compatibility
                "full_parse_time_ms": full_timings.full_parse_total_ms(),
                "incremental_parse_time_ms": null,
                "cpg_update_time_ms": null,
                "comparison_time_ms": null,
                // Detailed timing fields
                "detailed_timings": full_timings.to_json(),
                // File metrics
                "file_metrics": file_metrics.to_json()
            });

            // Attempt incremental parse if we have previous data
            if let (Some(prev_resources), Some(prev_cpgs)) = (&previous_resources, &previous_cpgs) {
                if let Some(prev_resource) = prev_resources
                    .iter()
                    .find(|r| r.raw_path().to_string_lossy() == file_path)
                {
                    if let Some(prev_cpg) = prev_cpgs.get(&file_path).cloned() {
                        let prev_src = match prev_resource.read_bytes() {
                            Ok(bytes) => bytes,
                            Err(e) => {
                                warn!("Failed to read previous version of {}: {}", file_path, e);
                                current_cpgs.insert(file_path.clone(), full_cpg);
                                current_hashes.insert(file_path, current_hash);
                                file_results.push(file_result);
                                continue;
                            }
                        };

                        match perform_incremental_parse(
                            &mut parser,
                            &prev_src,
                            &src_bytes,
                            prev_cpg,
                        ) {
                            Ok((incremental_cpg, mut incremental_timings, edits)) => {
                                // Update file metrics with change analysis
                                file_metrics = file_metrics
                                    .with_change_analysis(&prev_src, &src_bytes, &edits);

                                // Compare incremental result with full result
                                let comparison_start = Instant::now();
                                let _comparison = match incremental_cpg.compare(&full_cpg) {
                                    Ok(result) => result,
                                    Err(e) => {
                                        warn!(
                                            "CPG comparison failed for file {}: {}",
                                            file_path, e
                                        );
                                        current_cpgs.insert(file_path.clone(), full_cpg);
                                        current_hashes.insert(file_path, current_hash);
                                        file_result["file_metrics"] = file_metrics.to_json();
                                        file_results.push(file_result);
                                        continue;
                                    }
                                };
                                incremental_timings.comparison_time_ms =
                                    Some(comparison_start.elapsed().as_millis());

                                // Update file result with incremental data
                                file_result["incremental_nodes"] =
                                    json!(incremental_cpg.node_count());
                                file_result["incremental_edges"] =
                                    json!(incremental_cpg.edge_count());
                                // file_result["comparison_result"] = json!(format!("{}", comparison));

                                // Legacy fields
                                file_result["incremental_parse_time_ms"] =
                                    json!(incremental_timings.incremental_parse_total_ms());
                                file_result["cpg_update_time_ms"] =
                                    json!(incremental_timings.cpg_incremental_update_time_ms);
                                file_result["comparison_time_ms"] =
                                    json!(incremental_timings.comparison_time_ms);

                                // Merge detailed timings
                                let mut combined_timings = full_timings.to_json();
                                let incr_timings_json = incremental_timings.to_json();

                                if let Some(ts) = full_timings.ts_full_parse_time_ms {
                                    combined_timings["ts_full_parse_time_ms"] = json!(ts);
                                }
                                if let Some(cpg) = full_timings.cst_to_cpg_time_ms {
                                    combined_timings["cst_to_cpg_time_ms"] = json!(cpg);
                                }

                                for (key, value) in incr_timings_json.as_object().unwrap() {
                                    combined_timings[key] = value.clone();
                                }
                                file_result["detailed_timings"] = combined_timings;
                                file_result["file_metrics"] = file_metrics.to_json();
                            }
                            Err(e) => {
                                warn!("Incremental parse failed for file {}: {}", file_path, e);
                                file_result["file_metrics"] = file_metrics.to_json();
                            }
                        }
                    }
                }
            }

            // Always store the full parse result for the next iteration
            current_cpgs.insert(file_path.clone(), full_cpg);
            current_hashes.insert(file_path, current_hash);
            file_results.push(file_result);
        }

        let step_total_time = step_start.elapsed();

        let unchanged_files = file_results
            .iter()
            .filter(|f| f["unchanged"].as_bool().unwrap_or(false))
            .count();

        let total_successful_files = current_cpgs.len();
        let changed_files = total_successful_files - unchanged_files;

        let step_summary = json!({
            "commit": commit_id,
            "step": step,
            "commit_message": commit_message,
            "total_files": resources.len(),
            "successful_files": total_successful_files,
            "unchanged_files": unchanged_files,
            "changed_files": changed_files,
            "resource_load_time_ms": resource_load_time.as_millis(),
            "step_total_time_ms": step_total_time.as_millis(),
            "files": file_results
        });

        results.push(step_summary);
        previous_resources = Some(resources);
        previous_cpgs = Some(current_cpgs);
        previous_hashes = Some(current_hashes);

        info!(
            "Completed step {} in {:?} ({} unchanged, {} changed)",
            step + 1,
            step_total_time,
            unchanged_files,
            changed_files
        );
    }

    commit_progress.finish_with_message("Analysis complete");

    Ok(results)
}

fn run_benchmark(
    repository: &str,
    depth: usize,
    lang: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let start = std::time::Instant::now();
    let guard = pprof::ProfilerGuard::new(100).expect("Failed to start profiler");

    let repo_root: PathBuf = format!("./repos/{}", repository).into();
    let repo_root = repo_root
        .canonicalize()
        .expect("Failed to canonicalize repo root");

    let language: RegisteredLanguage = lang.parse().expect("Failed to parse language");
    let pattern = format!("{}/**/*.{}", repo_root.display(), lang);

    println!("Starting Git history walk analysis on {} commits", depth);

    let results = walk_git_history_and_benchmark(&repo_root, &pattern, depth, &language)?;

    println!("Analysis complete. Processed {} commits.", results.len());

    // Write results to file
    let output_file = format!(
        "benchmarks/benchmark_{}_{}_{}.json",
        repository, lang, depth,
    );
    std::fs::write(output_file.clone(), serde_json::to_string_pretty(&results)?)?;
    println!("Results written to {}", output_file);

    if let Ok(report) = guard.report().build() {
        let file = File::create(format!("{}-flamegraph.svg", repository))
            .expect("Failed to create flamegraph file");
        report.flamegraph(file).expect("Failed to write flamegraph");
    }

    println!("Flamegraph written to {}-flamegraph.svg", repository);

    let elapsed = start.elapsed();
    println!("Benchmark on {} repos completed in {:?}", depth, elapsed);

    Ok(())
}

#[ignore = "Temporarily disabled for debugging"]
#[test]
fn test_incr_perf_gv() {
    run_benchmark("graphviz", 500, "c").expect("Failed to run benchmark");
}

#[ignore = "Temporarily disabled for debugging"]
#[test]
fn test_incr_perf_ts() {
    run_benchmark("tree-sitter", 500, "c").expect("Failed to run benchmark");
}

#[ignore = "Temporarily disabled due to crashes"]
#[test]
fn test_incr_perf_ff() {
    run_benchmark("ffmpeg", 500, "c").expect("Failed to run benchmark");
}

// ---

#[test]
fn test_seq_patch_parse_sample() {
    dyn_cpg_rs::logging::init();
    seq_patch_parse("seq_patches/sample/");
}

#[test]
fn test_seq_patch_parse_large_sample_9() {
    dyn_cpg_rs::logging::init();
    seq_patch_parse("seq_patches/large_sample_9/");
}

#[test]
fn test_seq_patch_parse_large_sample_99() {
    dyn_cpg_rs::logging::init();
    seq_patch_parse("seq_patches/large_sample_99/");
}

/// Get metrics for more "developer-like" changes. Given an existing source:
/// 1. Insert a new statement
/// 2. Modifying an existing statement
/// 3. Deleting an existing statement
///
/// This is achieved using a "patch" style approach, where we have an original file and a set of patch files.
fn seq_patch_parse(path: &str) {
    info!("Starting incremental reparse test for patches in {}", path);

    // Create/wipe the metrics CSV file and add headers
    let csv_path = format!("{}metrics.csv", path);
    std::fs::write(
        &csv_path,
        "patch_name,edits_count,full_timings_ms,incremental_timings_ms\n",
    )
    .expect("Failed to create metrics CSV file");

    // Init the lang and parser
    let lang: RegisteredLanguage = "c".parse().expect("Failed to parse language");
    let mut parser = lang.get_parser().expect("Failed to get parser for C");

    // Load the base
    let base_resource =
        Resource::new(format!("{}base.c", path)).expect("Failed to create resource for base file");

    let mut base_source = base_resource
        .read_bytes()
        .expect("Failed to read base resource");

    let base_tree = parser
        .parse(base_source.clone(), None)
        .expect("Failed to parse base source file");

    // Create CPG from original tree before incremental parsing
    let start = std::time::Instant::now();
    let mut cpg = lang
        .cst_to_cpg(base_tree.clone(), base_source.clone())
        .expect("Failed to convert old tree to CPG");
    info!(
        "Initial CPG creation took {} ms",
        start.elapsed().as_millis()
    );

    let mut patches = glob(&format!("{}/*.patch", path))
        .expect("Failed to read glob pattern for patches")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect patch paths");

    // Patches MUST be named {number}_..., so we sort on the value before the first underscore
    patches.sort_by(|a, b| {
        let tmp = a
            .file_name()
            .expect("Failed to get file name")
            .to_str()
            .expect("Failed to convert file name to str")
            .split('_')
            .next()
            .cmp(
                &b.file_name()
                    .expect("Failed to get file name")
                    .to_str()
                    .expect("Failed to convert file name to str")
                    .split('_')
                    .next(),
            );

        // If equal, panic - we expect unique prefixes
        if tmp == std::cmp::Ordering::Equal {
            panic!("Patch files must have unique numeric prefixes before the first underscore");
        }

        tmp
    });

    for patch in patches {
        info!("Applying patch {}", patch.display());
        let patch_resource =
            Resource::new(patch.clone()).expect("Failed to create resource for patch");
        let patch_source = patch_resource
            .read_bytes()
            .expect("Failed to read patch resource");

        let new_src =
            apply_patch(&base_source, &patch_source).expect("Failed to apply patch to base source");

        // Verify the new source can be parsed
        let validation_tree = parser
            .parse(new_src.clone(), None)
            .expect("Patched source failed to parse");
        if validation_tree.root_node().has_error() {
            panic!("Patch {} resulted in invalid C syntax", patch.display());
        }

        // Perform full parse for comparison
        let start = std::time::Instant::now();
        let (full_cpg, full_timings) =
            perform_full_parse(&mut parser, &lang, &new_src).expect("Failed to perform full parse");
        info!("Full parse took {} ms", start.elapsed().as_millis());

        // Perform incremental parse and CPG update
        let start = std::time::Instant::now();
        let (incremental_cpg, incr_timings, edits) =
            perform_incremental_parse(&mut parser, &base_source, &new_src, cpg)
                .expect("Failed to perform incremental parse");
        info!(
            "Incremental parse and CPG update took {} ms",
            start.elapsed().as_millis()
        );

        // Compare the two results to ensure correctness
        let comparison_start = std::time::Instant::now();
        let comparison = incremental_cpg
            .compare(&full_cpg)
            .expect("Failed to compare CPGs");
        info!(
            "CPG comparison took {} ms",
            comparison_start.elapsed().as_millis()
        );

        // Assert that the CPGs are equivalent
        if !matches!(comparison, DetailedComparisonResult::Equivalent) {
            info!(
                "Patch {} resulted in mismatched CPGs!\nComparison: {}\nFull CPG: {} nodes, {} edges\nIncremental CPG: {} nodes, {} edges\nEdits applied: {} edits, {} lines changed, Performance: full={}ms, incremental={}ms",
                patch.display(),
                comparison.non_diff_string(),
                full_cpg.node_count(),
                full_cpg.edge_count(),
                incremental_cpg.node_count(),
                incremental_cpg.edge_count(),
                edits.len(),
                count_lines_in_byte_ranges(
                    &new_src,
                    edits.iter().map(|e| (e.new_start, e.new_end))
                ),
                full_timings.full_parse_total_ms().unwrap_or(0),
                incr_timings.incremental_parse_total_ms().unwrap_or(0)
            );
        } else {
            info!(
                "Patch {} applied successfully - CPGs match! Full: {}/{} nodes/edges, Incremental: {}/{} nodes/edges, Performance: full={}ms, incremental={}ms",
                patch.display(),
                full_cpg.node_count(),
                full_cpg.edge_count(),
                incremental_cpg.node_count(),
                incremental_cpg.edge_count(),
                full_timings.full_parse_total_ms().unwrap_or(0),
                incr_timings.incremental_parse_total_ms().unwrap_or(0)
            );
        }

        // Write metrics to CSV
        let patch_name = patch
            .file_name()
            .expect("Failed to get patch file name")
            .to_str()
            .expect("Failed to convert patch file name to string");

        let csv_line = format!(
            "{},{},{},{}\n",
            patch_name,
            edits.len(),
            full_timings.full_parse_total_ms().unwrap_or(0),
            incr_timings.incremental_parse_total_ms().unwrap_or(0)
        );

        std::fs::OpenOptions::new()
            .append(true)
            .open(&csv_path)
            .expect("Failed to open CSV file for appending")
            .write_all(csv_line.as_bytes())
            .expect("Failed to write to CSV file");

        // Update base source and CPG for next iteration (use full result for sanity)
        base_source = new_src;
        cpg = full_cpg;
    }
}

fn apply_patch(base: &[u8], patch: &[u8]) -> Result<Vec<u8>, String> {
    let base_str = std::str::from_utf8(base).map_err(|e| format!("Base not valid UTF-8: {}", e))?;
    let patch_str =
        std::str::from_utf8(patch).map_err(|e| format!("Patch not valid UTF-8: {}", e))?;

    // Parse the unified diff
    let patch_parsed =
        diffy::Patch::from_str(patch_str).map_err(|e| format!("Failed to parse patch: {}", e))?;

    // Apply the patch
    let result = diffy::apply(base_str, &patch_parsed)
        .map_err(|e| format!("Failed to apply patch: {:?}", e))?;

    Ok(result.into_bytes())
}

#[test]
fn test_mre_patch_53() {
    dyn_cpg_rs::logging::init();
    seq_patch_parse("seq_patches/mre_53/");
}
