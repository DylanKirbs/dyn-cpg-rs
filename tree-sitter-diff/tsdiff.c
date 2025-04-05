#include "meyers_sequence_matcher.h"
#include "test_loader.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tree_sitter/api.h>
#include <tree_sitter/tree-sitter-c.h>

void print_changed_ranges(TSRange *ranges, unsigned int count, char *source)
{

	unsigned int i, j, end, source_length = strlen(source);
	if (count == 0) {
		printf("No changes detected.\n");
		return;
	}

	for (i = 0; i < count; i++) {
		printf("Range %u: ", i);
		printf("Start byte: %u, ", ranges[i].start_byte);
		printf("End byte: %u, ", ranges[i].end_byte);
		printf("Start point: (%u, %u), ", ranges[i].start_point.row,
		       ranges[i].start_point.column);
		printf("End point: (%u, %u), ", ranges[i].end_point.row,
		       ranges[i].end_point.column);
		printf("Changed code: ");

		/* Add bounds checking */
		end = ranges[i].end_byte;
		if (end > source_length) {
			end = source_length;
		}

		for (j = ranges[i].start_byte; j < end; j++) {
			putchar(source[j]);
		}
		putchar('\n');
	}
}

/**
 * Print a side-by-side diff of two code samples
 *
 * @param original The original code string
 * @param original_length Length of the original code
 * @param modified The modified code string
 * @param modified_length Length of the modified code
 * @param segments Array of diff segments
 * @param segment_count Number of segments in the array
 */
void print_side_by_side_diff(const char *original,
                             const char *modified,
                             MSMDiffSegment *segments,
                             int segment_count)
{
	size_t len, j;
	int i;
	MSMDiffSegment segment;
	char diff_type, *original_text, *modified_text;
	char *diff_fmt = "[%3d:%3d] %-40.40s %c [%3d:%3d] %-40.40s\n";

	for (i = 0; i < segment_count; i++) {
		segment = segments[i];

		/* Determine diff symbol */
		switch (segment.type) {
			case INSERT:
				diff_type = '+';
				break;
			case DELETE:
				diff_type = '-';
				break;
			case REPLACE:
				diff_type = '|';
				break;
			case EQUAL:
				diff_type = ' ';
				break;
			default:
				diff_type = '?';
				break;
		}

		/* Extract the relevant portions of text */
		original_text = NULL;
		modified_text = NULL;

		/* For original text */
		if (segment.type != INSERT) {
			len = segment.end_a - segment.start_a;
			original_text = (char *) malloc(len + 1);
			if (original_text) {
				strncpy(original_text, original + segment.start_a, len);
				original_text[len] = '\0';
			}
		}

		/* For modified text */
		if (segment.type != DELETE) {
			size_t len = segment.end_b - segment.start_b;
			modified_text = (char *) malloc(len + 1);
			if (modified_text) {
				strncpy(modified_text, modified + segment.start_b, len);
				modified_text[len] = '\0';
			}
		}

		/* Print the diff, accounting for newlines */
		if (original_text) {
			for (j = 0; j < strlen(original_text); j++) {
				if (original_text[j] == '\n') {
					original_text[j] = ' ';
				}
			}
		}

		if (modified_text) {
			for (j = 0; j < strlen(modified_text); j++) {
				if (modified_text[j] == '\n') {
					modified_text[j] = ' ';
				}
			}
		}

		printf(diff_fmt, segment.start_a, segment.end_a,
		       original_text ? original_text : "", diff_type, segment.start_b,
		       segment.end_b, modified_text ? modified_text : "");

		/* Free allocated memory */
		free(original_text);
		free(modified_text);
	}
}

/**
 * Parse the old and new source allocating a tree for each.
 */
bool parse_and_update(const char *old_source,
                      const char *new_source,
                      TSParser *parser,
                      TSTree **old_tree,
                      TSTree **new_tree)
{
	TSInputEdit edit;
	MSMDiffResult *result;
	MSMDiffSegment segment;
	int i;

	if (!old_source || !new_source || !parser) {
		return false;
	}

	/* Old parse tree */
	*old_tree =
	    ts_parser_parse_string(parser, NULL, old_source, strlen(old_source));
	if (!*old_tree) {
		return false;
	}

	/* Textual Diff to update tree */
	result = msm_diff(old_source, new_source);
	if (!result) { /* No updates */
		*new_tree = *old_tree;
		return true;
	}

	for (i = 0; i < result->count; i++) {
		segment = result->segments[i];
		if (segment.type == EQUAL) {
			continue; /* Skip equal segments */
		}
		edit.start_byte = segment.start_a;
		edit.old_end_byte = segment.end_a;
		edit.new_end_byte = segment.end_b;
		ts_tree_edit(*old_tree, &edit);
	}

	msm_free_diff_result(result);

	/* Parse the new tree incrementally */
	*new_tree = ts_parser_parse_string(parser, *old_tree, new_source,
	                                   strlen(new_source));
	if (!*new_tree) {
		ts_tree_delete(*old_tree);
		return false;
	}

	return true;
}

bool cmp_expected_output(TSRange *ranges,
                         size_t range_count,
                         const char *exp_out)
{
	char line[128];
	size_t i = 0;
	const char *p = exp_out;
	uint32_t start_byte, end_byte, srow, scol, erow, ecol;
	TSRange r;

	while (*p && i < range_count) {
		/* Extract one line */
		size_t len = 0;
		while (p[len] && p[len] != '\n')
			len++;
		/* TODO: Note that this (v) may be problematic */
		if (len >= sizeof(line)) return false;

		memcpy(line, p, len);
		line[len] = '\0';
		p += len;
		if (*p == '\n') p++;

		/* Parse the line */
		if (sscanf(line, "{%u,%u,(%u,%u),(%u,%u)}", &start_byte, &end_byte,
		           &srow, &scol, &erow, &ecol) != 6) {
			return false;
		}

		/* Compare to the corresponding TSRange */
		r = ranges[i];
		if (r.start_byte != start_byte || r.end_byte != end_byte ||
		    r.start_point.row != srow || r.start_point.column != scol ||
		    r.end_point.row != erow || r.end_point.column != ecol) {
			return false;
		}

		i++;
	}

	/* Extra ranges or lines? */
	if (i != range_count) return false;
	if (*p != '\0') return false;

	return true;
}

int main()
{
	TSParser *parser;
	TSTree *old_tree = NULL, *new_tree = NULL;
	TSRange *ranges;
	unsigned int range_count;
	size_t i, j;
	TestSuite *suite;
	TestCase *test_case;
	bool parse_result;

	TestSuites suites = load_test_suites("tests");
	if (suites.suites == NULL || suites.count == 0) {
		printf("Failed to load suites\n");
		return 1;
	}

	/* Create a parser */
	parser = ts_parser_new();
	ts_parser_set_language(parser, tree_sitter_c());
	ts_parser_set_included_ranges(parser, NULL, 0);

	for (i = 0; i < suites.count; i++) {

		suite = &suites.suites[i];
		printf("Test Suite %lu:\n", i + 1);

		for (j = 0; j < suite->count; j++) {

			test_case = &suite->cases[j];
			if (strcmp(test_case->language, "c") != 0) continue;

			/* Print case */
			printf("Testing case: %s\n", test_case->name);

			parse_result = parse_and_update(test_case->original_source,
			                                test_case->modified_source, parser,
			                                &old_tree, &new_tree);
			if (!parse_result) {
				printf("FAILED TO PROCESS TREES\n");
				continue;
			}

			/* Get the changed ranges */
			ranges =
			    ts_tree_get_changed_ranges(old_tree, new_tree, &range_count);

			if (!cmp_expected_output(ranges, range_count,
			                         test_case->expected_output)) {
				printf("FAIL: Output Mismatch\n");
			}
			printf("PASS\n");

			/* Free resources */
			free(ranges);
			ts_tree_delete(old_tree);
			ts_tree_delete(new_tree);
		}
	}

	ts_parser_delete(parser);

	return 0;
}
