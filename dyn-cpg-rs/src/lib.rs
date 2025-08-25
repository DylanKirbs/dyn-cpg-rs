pub mod cpg;
pub mod languages;
pub mod resource;

pub mod logging {
    use tracing_subscriber::EnvFilter;

    pub fn init() {
        tracing_subscriber::fmt()
            .with_env_filter(EnvFilter::from_default_env())
            .init();
    }
}

pub mod diff {
    use similar::{capture_diff_slices, Algorithm, DiffOp};
    use tracing::debug;
    use tree_sitter::{Parser, Point, Tree};

    #[derive(Debug, Clone)]
    pub struct SourceEdit {
        pub old_start: usize,
        pub old_end: usize,
        pub new_start: usize,
        pub new_end: usize,
    }

    struct LineIndex {
        line_starts: Vec<usize>,
    }

    impl LineIndex {
        fn new(buf: &[u8]) -> Self {
            let mut line_starts = vec![0];
            for (i, b) in buf.iter().enumerate() {
                if *b == b'\n' {
                    line_starts.push(i + 1);
                }
            }
            Self { line_starts }
        }

        fn point_for_offset(&self, offset: usize) -> Point {
            match self.line_starts.binary_search(&offset) {
                Ok(row) => Point { row, column: 0 },
                Err(row) => {
                    let row = row.saturating_sub(1);
                    let col = offset - self.line_starts[row];
                    Point { row, column: col }
                }
            }
        }
    }

    fn source_edits(old: &[u8], new: &[u8]) -> Vec<SourceEdit> {
        let diff_ops = capture_diff_slices(Algorithm::Myers, old, new);
        let mut edits = Vec::new();

        for op in diff_ops {
            match op {
                DiffOp::Equal { .. } => continue,
                DiffOp::Insert {
                    old_index,
                    new_index,
                    new_len,
                } => {
                    edits.push(SourceEdit {
                        old_start: old_index,
                        old_end: old_index,
                        new_start: new_index,
                        new_end: new_index + new_len,
                    });
                }
                DiffOp::Delete {
                    old_index,
                    old_len,
                    new_index,
                } => {
                    edits.push(SourceEdit {
                        old_start: old_index,
                        old_end: old_index + old_len,
                        new_start: new_index,
                        new_end: new_index,
                    });
                }
                DiffOp::Replace {
                    old_index,
                    old_len,
                    new_index,
                    new_len,
                } => {
                    edits.push(SourceEdit {
                        old_start: old_index,
                        old_end: old_index + old_len,
                        new_start: new_index,
                        new_end: new_index + new_len,
                    });
                }
            }
        }

        edits
    }

    pub fn incremental_parse(
        parser: &mut Parser,
        old_src: &[u8],
        new_src: &[u8],
        old_tree: &mut Tree,
    ) -> Result<(Vec<SourceEdit>, Tree), String> {
        debug!(
            "Incremental parsing from old source of length {} to new source of length {}",
            old_src.len(),
            new_src.len()
        );

        let line_index = LineIndex::new(old_src);
        let new_line_index = LineIndex::new(new_src);

        let edits = source_edits(old_src, new_src);
        debug!("Applying {} edits", edits.len());

        for edit in edits.clone() {
            let start_point = line_index.point_for_offset(edit.old_start);
            let old_end_point = line_index.point_for_offset(edit.old_end);
            let new_end_point = new_line_index.point_for_offset(edit.new_end);

            old_tree.edit(&tree_sitter::InputEdit {
                start_byte: edit.old_start,
                old_end_byte: edit.old_end,
                new_end_byte: edit.new_end,
                start_position: start_point,
                old_end_position: old_end_point,
                new_end_position: new_end_point,
            });
        }

        debug!("Parsing with updated edits");
        let tree = parser
            .parse(new_src, Some(old_tree))
            .ok_or_else(|| "Failed to parse new source".to_string())?;

        debug!("Incremental parse completed successfully");
        Ok((edits, tree))
    }
}
