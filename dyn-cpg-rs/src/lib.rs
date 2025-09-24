pub mod cpg;
pub mod languages;
pub mod resource;

pub mod logging {
    use tracing_subscriber::EnvFilter;

    pub fn init() {
        static LOGGING_INIT: std::sync::Once = std::sync::Once::new();
        LOGGING_INIT.call_once(|| {
            tracing_subscriber::fmt()
                .with_env_filter(EnvFilter::from_default_env())
                .init();
        });
    }
}

pub mod diff {
    use similar::{Algorithm, DiffOp, capture_diff_slices};
    use tracing::{debug, trace};
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
        // Helper: split s into alternating runs of whitespace / non-whitespace and record byte offsets.
        fn tokenize_with_offsets(s: &str) -> (Vec<&str>, Vec<usize>) {
            let mut tokens = Vec::new();
            let mut offsets = Vec::new();
            let mut seg_start = 0usize;
            let mut last_is_ws: Option<bool> = None;

            for (idx, ch) in s.char_indices() {
                let is_ws = ch.is_whitespace();
                if let Some(last) = last_is_ws {
                    if is_ws != last {
                        tokens.push(&s[seg_start..idx]);
                        offsets.push(seg_start);
                        seg_start = idx;
                        last_is_ws = Some(is_ws);
                    }
                } else {
                    last_is_ws = Some(is_ws);
                }
            }

            if seg_start < s.len() {
                tokens.push(&s[seg_start..]);
                offsets.push(seg_start);
            }

            (tokens, offsets)
        }

        // Map token index + token count to byte range in the original string.
        fn token_range_to_byte_range(
            offsets: &[usize],
            tokens: &[&str],
            index: usize,
            len: usize,
            total_bytes: usize,
        ) -> (usize, usize) {
            if len == 0 {
                // insertion at token boundary
                if index >= offsets.len() {
                    (total_bytes, total_bytes)
                } else {
                    let start = offsets[index];
                    (start, start)
                }
            } else {
                let start = offsets[index];
                let last = index + len - 1;
                let end = offsets[last] + tokens[last].len();
                (start, end)
            }
        }

        // Try UTF-8 tokenized (word-aware) diff first.
        if let (Ok(old_s), Ok(new_s)) = (std::str::from_utf8(old), std::str::from_utf8(new)) {
            let (old_tokens, old_offsets) = tokenize_with_offsets(old_s);
            let (new_tokens, new_offsets) = tokenize_with_offsets(new_s);

            let diff_ops = capture_diff_slices(Algorithm::Myers, &old_tokens, &new_tokens);
            let mut edits = Vec::new();

            for op in diff_ops {
                match op {
                    DiffOp::Equal { .. } => continue,
                    DiffOp::Insert {
                        old_index,
                        new_index,
                        new_len,
                    } => {
                        let (old_start, old_end) = token_range_to_byte_range(
                            &old_offsets,
                            &old_tokens,
                            old_index,
                            0,
                            old_s.len(),
                        );
                        let (new_start, new_end) = token_range_to_byte_range(
                            &new_offsets,
                            &new_tokens,
                            new_index,
                            new_len,
                            new_s.len(),
                        );
                        edits.push(SourceEdit {
                            old_start,
                            old_end,
                            new_start,
                            new_end,
                        });
                    }
                    DiffOp::Delete {
                        old_index,
                        old_len,
                        new_index,
                    } => {
                        let (old_start, old_end) = token_range_to_byte_range(
                            &old_offsets,
                            &old_tokens,
                            old_index,
                            old_len,
                            old_s.len(),
                        );
                        let (new_start, new_end) = token_range_to_byte_range(
                            &new_offsets,
                            &new_tokens,
                            new_index,
                            0,
                            new_s.len(),
                        );
                        edits.push(SourceEdit {
                            old_start,
                            old_end,
                            new_start,
                            new_end,
                        });
                    }
                    DiffOp::Replace {
                        old_index,
                        old_len,
                        new_index,
                        new_len,
                    } => {
                        let (old_start, old_end) = token_range_to_byte_range(
                            &old_offsets,
                            &old_tokens,
                            old_index,
                            old_len,
                            old_s.len(),
                        );
                        let (new_start, new_end) = token_range_to_byte_range(
                            &new_offsets,
                            &new_tokens,
                            new_index,
                            new_len,
                            new_s.len(),
                        );
                        edits.push(SourceEdit {
                            old_start,
                            old_end,
                            new_start,
                            new_end,
                        });
                    }
                }
            }

            return edits;
        }

        // Fallback: byte-level diff (original behaviour) for non-UTF-8.
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

    pub fn incremental_ts_parse(
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
        debug!("Applying edits: {:#?}", edits);

        for (i, edit) in edits.clone().into_iter().enumerate() {
            let start_point = line_index.point_for_offset(edit.old_start);
            let old_end_point = line_index.point_for_offset(edit.old_end);
            let new_end_point = new_line_index.point_for_offset(edit.new_end);

            let new_end_byte = (edit.old_start + edit.new_end).saturating_sub(edit.new_start);

            let ts_edit = tree_sitter::InputEdit {
                start_byte: edit.old_start,
                old_end_byte: edit.old_end,
                new_end_byte,
                start_position: start_point,
                old_end_position: old_end_point,
                new_end_position: new_end_point,
            };

            trace!("Applying tree-sitter edit {}: {:?}", i, ts_edit);
            old_tree.edit(&ts_edit);
        }

        debug!("Parsing with updated edits");
        let tree = parser
            .parse(new_src, Some(old_tree))
            .ok_or_else(|| "Failed to parse new source".to_string())?;

        debug!("Incremental parse completed successfully");
        Ok((edits, tree))
    }
}
