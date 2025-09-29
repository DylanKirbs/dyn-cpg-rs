use dyn_cpg_rs::{
    cpg::{Cpg, DetailedComparisonResult},
    languages::RegisteredLanguage,
    resource::Resource,
};
use git2::{Oid, Repository};
use glob::glob;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tracing::{info, warn};

#[derive(Default, Debug)]
struct PatchResult {
    patch_name: String,
    same: u8, // 1 = same, 0 = different
    full_node_count: usize,
    full_edge_count: usize,
    incr_node_count: usize,
    incr_edge_count: usize,
    edits_count: usize,
    lines_changed: usize,
    full_time_ms: u128,
    incr_time_ms: u128,
    file_size_bytes: usize,
    commit_hash: String,
    file_path: String,
}

impl PatchResult {
    pub fn to_csv_header(&self) -> String {
        "patch_name,edits_count,full_timings_ms,incremental_timings_ms,same,full_nodes,full_edges,incremental_nodes,incremental_edges,lines_changed,file_size_bytes,commit_hash,file_path\n".to_string()
    }

    pub fn to_csv_line(&self) -> String {
        format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            self.patch_name,
            self.edits_count,
            self.full_time_ms,
            self.incr_time_ms,
            self.same,
            self.full_node_count,
            self.full_edge_count,
            self.incr_node_count,
            self.incr_edge_count,
            self.lines_changed,
            self.file_size_bytes,
            self.commit_hash,
            self.file_path
        )
    }

    pub fn display(&self) -> String {
        format!(
            "Patch {}: same={}, full_nodes={}, full_edges={}, incr_nodes={}, incr_edges={}, edits_count={}, lines_changed={}, full_time_ms={}, incr_time_ms={}, file_size={}, commit={}, file={}",
            self.patch_name,
            self.same,
            self.full_node_count,
            self.full_edge_count,
            self.incr_node_count,
            self.incr_edge_count,
            self.edits_count,
            self.lines_changed,
            self.full_time_ms,
            self.incr_time_ms,
            self.file_size_bytes,
            self.commit_hash,
            self.file_path
        )
    }

    pub fn log(&self) {
        if self.same == 1 {
            info!("{}", self.display());
        } else {
            warn!("{}", self.display());
        }
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

fn perform_full_parse(
    parser: &mut tree_sitter::Parser,
    lang: &RegisteredLanguage,
    src_bytes: &[u8],
) -> Result<(Cpg, u128, u128), String> {
    // Step 1: Tree-sitter parse
    let ts_start = Instant::now();
    let tree = parser
        .parse(src_bytes, None)
        .ok_or("Tree-sitter parse failed")?;

    // Step 2: CST → CPG conversion
    let cst_start = Instant::now();
    let cpg = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        lang.cst_to_cpg(tree, src_bytes.to_vec())
    }))
    .map_err(|_| "CST to CPG panicked (likely stack overflow)")?
    .map_err(|e| format!("CST to CPG failed: {}", e))?;

    Ok((
        cpg,
        ts_start.elapsed().as_millis(),
        cst_start.elapsed().as_millis(),
    ))
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

// --- Walk Patches --- //

/// Get metrics for more "developer-like" changes. Given an existing source:
/// 1. Insert a new statement
/// 2. Modifying an existing statement
/// 3. Deleting an existing statement
///
/// This is achieved using a "patch" style approach, where we have an original file and a set of patch files.
fn walk_patches(path: &str) {
    info!("Starting incremental reparse test for patches in {}", path);

    // Create/wipe the metrics CSV file and add headers
    let csv_path = format!("{}metrics.csv", path);
    std::fs::write(&csv_path, PatchResult::default().to_csv_header())
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
    let mut cpg = lang
        .cst_to_cpg(base_tree.clone(), base_source.clone())
        .expect("Failed to convert old tree to CPG");

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
        let (full_cpg, full_ts_ms, _) =
            perform_full_parse(&mut parser, &lang, &new_src).expect("Failed to perform full parse");

        // Perform incremental parse and CPG update
        let incr_metrics = cpg
            .incremental_update(&mut parser, new_src.clone())
            .expect("Failed to perform incremental parse and CPG update");

        // Compare the two results to ensure correctness
        let comparison_time = Instant::now();
        let comparison = cpg.compare(&full_cpg).expect("Failed to compare CPGs");
        let comparison_duration = comparison_time.elapsed();
        info!("CPG comparison completed in {:?}", comparison_duration);

        let res = PatchResult {
            patch_name: patch
                .file_name()
                .expect("Failed to get patch file name")
                .to_str()
                .expect("Failed to convert patch file name to string")
                .to_string(),
            same: if matches!(comparison, DetailedComparisonResult::Equivalent) {
                1
            } else {
                0
            },
            full_node_count: full_cpg.node_count(),
            full_edge_count: full_cpg.edge_count(),
            incr_node_count: cpg.node_count(),
            incr_edge_count: cpg.edge_count(),
            edits_count: incr_metrics.edits.clone().unwrap_or_default().len(),
            lines_changed: count_lines_in_byte_ranges(
                &new_src,
                incr_metrics
                    .edits
                    .clone()
                    .unwrap_or_default()
                    .iter()
                    .map(|e| (e.new_start, e.new_end)),
            ),
            full_time_ms: full_ts_ms,
            incr_time_ms: incr_metrics.total_time_ms.unwrap_or(0),
            file_size_bytes: new_src.len(),
            commit_hash: "patch".to_string(), // Default for patch-based metrics
            file_path: "base.c".to_string(),  // Default for patch-based metrics
        };
        res.log();

        std::fs::OpenOptions::new()
            .append(true)
            .open(&csv_path)
            .expect("Failed to open CSV file for appending")
            .write_all(res.to_csv_line().as_bytes())
            .expect("Failed to write to CSV file");

        // Update base source and CPG for next iteration (use full result for sanity)
        base_source = new_src;
        cpg = full_cpg;
    }
}

// --- Walk Commits --- //

/// Walk through git commits and analyze incremental parsing performance.
/// Similar to walk_patches but uses git history instead of patch files.
fn walk_git_commits(repo_name: &str, file_pattern: &str, max_commits: usize) {
    info!(
        "Starting incremental reparse test for git history in repos/{}",
        repo_name
    );

    // Start profiling
    let guard = pprof::ProfilerGuard::new(100).unwrap(); // Sample at 100Hz
    let start_time = Instant::now();

    // Create/wipe the metrics CSV file and add headers
    let csv_path = format!("repos/metrics/{}-{}.csv", repo_name, max_commits);
    std::fs::write(&csv_path, PatchResult::default().to_csv_header())
        .expect("Failed to create metrics CSV file");

    // Init the lang
    let lang: RegisteredLanguage = "c".parse().expect("Failed to parse language");

    // Open the git repository
    let repo_path = format!("repos/{}", repo_name);
    let repo = Repository::open(&repo_path).expect("Failed to open git repository");

    // Get commit history
    let mut revwalk = repo.revwalk().expect("Failed to create revwalk");
    revwalk.push_head().expect("Failed to push HEAD");

    let mut commits: Vec<Oid> = revwalk
        .take(max_commits + 1) // +1 to have a previous commit for comparison
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect commits");

    commits.reverse(); // Process from oldest to newest

    if commits.len() < 2 {
        warn!("Not enough commits found for comparison");
        return;
    }

    // Store previous state for incremental parsing
    let mut previous_cpgs: HashMap<String, Cpg> = HashMap::new();
    let mut previous_sources: HashMap<String, Vec<u8>> = HashMap::new();
    let mut files_processed = 0;
    let total_commits = commits.len() - 1;

    println!("{}: processing {} commits", repo_name, total_commits);

    for (i, &commit_oid) in commits.iter().enumerate().skip(1) {
        let commit = repo.find_commit(commit_oid).expect("Failed to find commit");
        let commit_hash = commit.id().to_string();

        // Progress indicator every 10 commits
        if i % 10 == 0 || i == commits.len() - 1 {
            println!(
                "{}: commit {}/{} ({}%) - {} entries recorded so far",
                repo_name,
                i,
                total_commits,
                (i * 100) / total_commits,
                files_processed
            );
        }

        info!(
            "Processing commit {} ({}/{})",
            &commit_hash[..8],
            i,
            total_commits
        );

        // Get the tree for this commit
        let tree = commit.tree().expect("Failed to get tree for commit");
        let mut c_files = Vec::new();

        tree.walk(git2::TreeWalkMode::PreOrder, |root, entry| {
            if let Some(name) = entry.name() {
                let full_path = if root.is_empty() {
                    name.to_string()
                } else {
                    format!("{}/{}", root, name)
                };

                // Simple pattern matching for C files (you could make this more sophisticated)
                if full_path.ends_with(".c") && full_path.contains(file_pattern) {
                    if let Some(object) = entry.to_object(&repo).ok() {
                        if let Some(blob) = object.as_blob() {
                            let current_source = blob.content().to_vec();

                            if !current_source.is_empty() {
                                c_files.push((full_path, current_source));
                            }
                        }
                    }
                }
            }
            git2::TreeWalkResult::Ok
        })
        .expect("Failed to walk tree");

        // Thread-safe containers for results
        let results = Arc::new(Mutex::new(Vec::new()));
        let updated_cpgs = Arc::new(Mutex::new(HashMap::new()));
        let updated_sources = Arc::new(Mutex::new(HashMap::new()));

        // Process files in parallel
        c_files.into_par_iter().for_each(|(full_path, current_source)| {
            // Check if we have previous data and if file has changed
            if let (Some(prev_cpg), Some(prev_source)) = (
                previous_cpgs.get(&full_path).cloned(),
                previous_sources.get(&full_path)
            ) {
                // Only process if the file actually changed
                if prev_source != &current_source {
                    let mut incr_cpg = prev_cpg;

                    // Create a parser for this thread
                    let mut parser = lang.get_parser().expect("Failed to get parser for C");

                    // Validate that the file can be parsed
                    if parser.parse(&current_source, None).is_none() {
                        warn!("Skipping file {} - failed to parse", full_path);
                        return;
                    }

                    // Perform full parse for reference
                    let (full_cpg, full_ts_ms, _) = match perform_full_parse(&mut parser, &lang, &current_source) {
                        Ok(result) => result,
                        Err(e) => {
                            warn!("Full parse failed for {}: {}", full_path, e);
                            return;
                        }
                    };

                    let mut result = PatchResult {
                        patch_name: format!("{}_{}", &commit_hash[..8], full_path.replace("/", "_")),
                        same: 0,
                        full_node_count: full_cpg.node_count(),
                        full_edge_count: full_cpg.edge_count(),
                        incr_node_count: 0,
                        incr_edge_count: 0,
                        edits_count: 0,
                        lines_changed: 0,
                        full_time_ms: full_ts_ms,
                        incr_time_ms: 0,
                        file_size_bytes: current_source.len(),
                        commit_hash: commit_hash.clone(),
                        file_path: full_path.clone(),
                    };

                    // Perform incremental update
                    match incr_cpg.incremental_update(&mut parser, current_source.clone()) {
                        Ok(incr_metrics) => {
                            // Compare incremental result with full result
                            let comparison = incr_cpg.compare(&full_cpg);

                            match comparison {
                                Ok(DetailedComparisonResult::Equivalent) => {
                                    result.same = 1;
                                }
                                Ok(_) => {
                                    result.same = 0;
                                    warn!("Incremental parsing result differs from full parse for {}", full_path);
                                }
                                Err(e) => {
                                    result.same = 3;
                                    warn!("Failed to compare CPGs for {}: {}", full_path, e);
                                }
                            }

                            result.incr_node_count = incr_cpg.node_count();
                            result.incr_edge_count = incr_cpg.edge_count();
                            result.edits_count = incr_metrics.edits.clone().unwrap_or_default().len();
                            result.lines_changed = count_lines_in_byte_ranges(
                                &current_source,
                                incr_metrics
                                    .edits
                                    .clone()
                                    .unwrap_or_default()
                                    .iter()
                                    .map(|e| (e.new_start, e.new_end)),
                            );
                            result.incr_time_ms = incr_metrics.total_time_ms.unwrap_or(0);

                            // Store results for writing later
                            results.lock().unwrap().push(result);
                            updated_cpgs.lock().unwrap().insert(full_path.clone(), full_cpg);
                            updated_sources.lock().unwrap().insert(full_path, current_source);
                        }
                        Err(e) => {
                            warn!("Incremental parsing failed for {}: {}", full_path, e);
                            result.same = 3;

                            // Still record the entry to track failures
                            results.lock().unwrap().push(result);
                            updated_cpgs.lock().unwrap().insert(full_path.clone(), full_cpg);
                            updated_sources.lock().unwrap().insert(full_path, current_source);
                        }
                    }
                }
            } else {
                // New file - establish baseline for it
                let mut parser = lang.get_parser().expect("Failed to get parser for C");

                if let Ok((full_cpg, full_ts_ms, _)) = perform_full_parse(&mut parser, &lang, &current_source) {
                    let result = PatchResult {
                        patch_name: format!("{}_{}", &commit_hash[..8], full_path.replace("/", "_")),
                        same: 2, // New file marker
                        full_node_count: full_cpg.node_count(),
                        full_edge_count: full_cpg.edge_count(),
                        incr_node_count: 0,
                        incr_edge_count: 0,
                        edits_count: 0,
                        lines_changed: 0,
                        full_time_ms: full_ts_ms,
                        incr_time_ms: 0,
                        file_size_bytes: current_source.len(),
                        commit_hash: commit_hash.clone(),
                        file_path: full_path.clone(),
                    };

                    results.lock().unwrap().push(result);
                    updated_cpgs.lock().unwrap().insert(full_path.clone(), full_cpg);
                    updated_sources.lock().unwrap().insert(full_path, current_source);
                }
            }
        });

        // Write all results to CSV sequentially to maintain order
        let results = Arc::try_unwrap(results).unwrap().into_inner().unwrap();
        for result in results {
            result.log();

            std::fs::OpenOptions::new()
                .append(true)
                .open(&csv_path)
                .expect("Failed to open CSV file for appending")
                .write_all(result.to_csv_line().as_bytes())
                .expect("Failed to write to CSV file");

            files_processed += 1;
        }

        // Update the baseline state for next iteration
        let updated_cpgs = Arc::try_unwrap(updated_cpgs).unwrap().into_inner().unwrap();
        let updated_sources = Arc::try_unwrap(updated_sources)
            .unwrap()
            .into_inner()
            .unwrap();

        previous_cpgs.extend(updated_cpgs);
        previous_sources.extend(updated_sources);
    }

    let total_duration = start_time.elapsed();

    info!(
        "Completed git history analysis for {} - processed {} files across {} commits in {:?}",
        repo_name, files_processed, total_commits, total_duration
    );

    // Generate profiling reports
    if let Ok(report) = guard.report().build() {
        // Generate flamegraph
        let flamegraph_path = format!("repos/metrics/{}-{}_flamegraph.svg", repo_name, max_commits);
        if let Ok(file) = File::create(&flamegraph_path) {
            if let Err(e) = report.flamegraph(file) {
                warn!("Failed to write flamegraph: {}", e);
            } else {
                info!("Flamegraph written to {}", flamegraph_path);
            }
        }

        // Generate text profile report
        let profile_path = format!("repos/metrics/{}-{}_profile.txt", repo_name, max_commits);
        if let Ok(mut file) = File::create(&profile_path) {
            writeln!(
                file,
                "Performance Profile for {} ({} commits, {} files processed)",
                repo_name, total_commits, files_processed
            )
            .unwrap();
            writeln!(file, "Total duration: {:?}", total_duration).unwrap();
            writeln!(file, "").unwrap();

            // Filter samples to only include stacks with relevant crate functions
            let mut filtered_samples: Vec<_> = report
                .data
                .iter()
                .filter(|(stack, _)| {
                    // Check if any frame in the stack contains functions from our crate or dependencies
                    stack.frames.iter().any(|frame| {
                        frame.iter().any(|sym| {
                            let name = sym.name();
                            let filename = sym.filename();
                            // Include functions from dyn-cpg-rs, tree-sitter, git2, rayon, etc.
                            name.contains("dyn_cpg_rs") ||
                            name.contains("tree_sitter") ||
                            name.contains("git2") ||
                            name.contains("rayon") ||
                            name.contains("pprof") ||
                            filename.contains("dyn-cpg-rs") ||
                            filename.contains("tree-sitter") ||
                            filename.contains("git2") ||
                            filename.contains("rayon") ||
                            // Also include Rust std functions that are directly called by our code
                            (name.starts_with("std::") && (
                                name.contains("parse") ||
                                name.contains("clone") ||
                                name.contains("collect") ||
                                name.contains("HashMap") ||
                                name.contains("Vec") ||
                                name.contains("String")
                            ))
                        })
                    })
                })
                .collect();

            filtered_samples.sort_by(|a, b| b.1.cmp(a.1));

            writeln!(
                file,
                "Top functions by sample count (filtered for relevant crate code):"
            )
            .unwrap();
            writeln!(
                file,
                "{:<80} {:<10} {:<10}",
                "Function", "Samples", "Percentage"
            )
            .unwrap();
            writeln!(file, "{}", "-".repeat(102)).unwrap();

            let total_samples: i64 = filtered_samples
                .iter()
                .map(|(_, count)| **count as i64)
                .sum();

            if total_samples == 0 {
                writeln!(
                    file,
                    "No relevant crate functions found in profile samples."
                )
                .unwrap();
                writeln!(file, "This might indicate the profiling sample rate is too low or the test ran too quickly.").unwrap();
            } else {
                for (i, (stack, count)) in filtered_samples.iter().take(20).enumerate() {
                    let percentage = (**count as f64 / total_samples as f64) * 100.0;

                    // Find the most relevant function name in the stack
                    let function_name = stack
                        .frames
                        .iter()
                        .filter_map(|frame| {
                            frame
                                .iter()
                                .find(|sym| {
                                    let name = sym.name();
                                    name.contains("dyn_cpg_rs")
                                        || name.contains("tree_sitter")
                                        || name.contains("git2")
                                })
                                .map(|sym| {
                                    let name = sym.name();

                                    let tmp_fname = format!("{}", sym.filename());
                                    let fname = tmp_fname.split('/').last().unwrap_or(&tmp_fname);

                                    let meta = format!(" ({}:{})", fname, sym.lineno());

                                    let short_name = if (name.len() + meta.len()) > 70 {
                                        // Try to extract just the function name from long type signatures
                                        if let Some(last_colon) = name.rfind("::") {
                                            let after_colon = &name[last_colon + 2..];
                                            if (after_colon.len() + meta.len()) < 50 {
                                                after_colon.to_string()
                                            } else {
                                                format!(
                                                    "...{}",
                                                    &after_colon[after_colon
                                                        .len()
                                                        .saturating_sub(
                                                            50_usize.saturating_sub(meta.len())
                                                        )..]
                                                )
                                            }
                                        } else {
                                            format!(
                                                "...{}",
                                                &name[name.len().saturating_sub(
                                                    70_usize.saturating_sub(meta.len())
                                                )..]
                                            )
                                        }
                                    } else {
                                        name.to_string()
                                    };
                                    format!("{}{}", short_name, meta)
                                })
                        })
                        .next()
                        .unwrap_or_else(|| {
                            // Fallback to the top frame if no specific crate function found
                            stack
                                .frames
                                .last()
                                .and_then(|frame| frame.first())
                                .map(|sym| {
                                    let name = sym.name();
                                    if name.len() > 70 {
                                        format!("...{}", &name[name.len() - 70..])
                                    } else {
                                        name.to_string()
                                    }
                                })
                                .unwrap_or_else(|| "<unknown>".to_string())
                        });

                    writeln!(
                        file,
                        "{:<80} {:<10} {:<10.2}%",
                        function_name.chars().take(79).collect::<String>(),
                        count,
                        percentage
                    )
                    .unwrap();

                    if i >= 19 {
                        break;
                    }
                }
            }

            writeln!(file, "").unwrap();
            writeln!(
                file,
                "Detailed stack traces for relevant functions (top 10):"
            )
            .unwrap();
            writeln!(file, "{}", "=".repeat(70)).unwrap();

            for (i, (stack, count)) in filtered_samples.iter().take(10).enumerate() {
                let percentage = (**count as f64 / total_samples as f64) * 100.0;
                writeln!(file, "{}. Samples: {} ({:.2}%)", i + 1, count, percentage).unwrap();
                writeln!(file, "   Stack trace (most relevant frames):").unwrap();

                // Show frames in reverse order (call stack), but prioritize relevant ones
                let mut relevant_frames = Vec::new();
                let mut other_frames = Vec::new();

                for frame in stack.frames.iter().rev() {
                    let has_relevant_sym = frame.iter().any(|sym| {
                        let name = sym.name();
                        let filename = sym.filename();
                        name.contains("dyn_cpg_rs")
                            || name.contains("tree_sitter")
                            || name.contains("git2")
                            || filename.contains("dyn-cpg-rs")
                    });

                    if has_relevant_sym {
                        relevant_frames.push(frame);
                    } else {
                        other_frames.push(frame);
                    }
                }

                // Show relevant frames first, then a few other frames for context
                let frame_context = 5;
                for frame in relevant_frames.iter().take(frame_context) {
                    for sym in frame.iter() {
                        let name = sym.name();
                        let filename = sym.filename();
                        let lineno = sym.lineno();

                        // Only show the most relevant symbol per frame
                        if name.contains("dyn_cpg_rs")
                            || name.contains("tree_sitter")
                            || name.contains("git2")
                        {
                            writeln!(
                                file,
                                "     -> {} ({}:{})",
                                name,
                                filename.split('/').last().unwrap_or(&filename),
                                lineno
                            )
                            .unwrap();
                            break; // Only show one symbol per frame to avoid clutter
                        }
                    }
                }

                // Add a few context frames if we have room
                if relevant_frames.len() < frame_context {
                    for frame in other_frames
                        .iter()
                        .take(frame_context - relevant_frames.len())
                    {
                        if let Some(sym) = frame.first() {
                            let name = sym.name();
                            let filename = sym.filename();
                            let lineno = sym.lineno();
                            writeln!(
                                file,
                                "     -> {} ({}:{})",
                                name,
                                filename.split('/').last().unwrap_or(&filename),
                                lineno
                            )
                            .unwrap();
                        }
                    }
                }

                writeln!(file, "").unwrap();
            }

            info!("Profile report written to {}", profile_path);
        }
    } else {
        warn!("Failed to build profiling report");
    }
}

// --- Git History Tests --- //

#[ignore = "Large repository - only run when needed"]
#[test]
fn test_git_history_graphviz() {
    dyn_cpg_rs::logging::init();
    walk_git_commits("graphviz", "", 500);
}

#[ignore = "Large repository - only run when needed"]
#[test]
fn test_git_history_tree_sitter() {
    dyn_cpg_rs::logging::init();
    walk_git_commits("tree-sitter", "", 500);
}

#[ignore = "Very large repository - only run when needed"]
#[test]
fn test_git_history_ffmpeg() {
    dyn_cpg_rs::logging::init();
    walk_git_commits("ffmpeg", "", 500);
}

#[test]
fn test_git_history_small() {
    dyn_cpg_rs::logging::init();
    walk_git_commits("tree-sitter", "", 10);
}

// --- Patch Tests --- //

#[test]
fn test_seq_patch_parse_sample() {
    dyn_cpg_rs::logging::init();
    walk_patches("seq_patches/sample/");
}

#[test]
fn test_seq_patch_parse_large_sample_9() {
    dyn_cpg_rs::logging::init();
    walk_patches("seq_patches/large_sample_9/");
}

#[ignore = "Quite slow, only run if we want the metrics"]
#[test]
fn test_seq_patch_parse_large_sample_99() {
    dyn_cpg_rs::logging::init();
    walk_patches("seq_patches/large_sample_99/");
}

#[test]
fn test_mre_seq_patch_53() {
    dyn_cpg_rs::logging::init();
    walk_patches("seq_patches/mre_53/");
}
