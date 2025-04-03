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

	unsigned int source_length = strlen(source);
	if (count == 0) {
		printf("No changes detected.\n");
		return;
	}

	for (unsigned int i = 0; i < count; i++) {
		printf("Range %u: ", i);
		printf("Start byte: %u, ", ranges[i].start_byte);
		printf("End byte: %u, ", ranges[i].end_byte);
		printf("Start point: (%u, %u), ", ranges[i].start_point.row,
		       ranges[i].start_point.column);
		printf("End point: (%u, %u), ", ranges[i].end_point.row,
		       ranges[i].end_point.column);
		printf("Changed code: ");

		// Add bounds checking
		unsigned int end = ranges[i].end_byte;
		if (end > source_length) {
			end = source_length;
		}

		for (unsigned int j = ranges[i].start_byte; j < end; j++) {
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
	char *diff_fmt = "[%3d:%3d] %-40.40s %c [%3d:%3d] %-40.40s\n";

	for (int i = 0; i < segment_count; i++) {
		MSMDiffSegment segment = segments[i];
		char diff_type = ' ';

		// Determine diff symbol
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
		}

		// Extract the relevant portions of text
		char *original_text = NULL;
		char *modified_text = NULL;

		// For original text
		if (segment.type != INSERT) {
			size_t len = segment.end_a - segment.start_a;
			original_text = (char *) malloc(len + 1);
			if (original_text) {
				strncpy(original_text, original + segment.start_a, len);
				original_text[len] = '\0';
			}
		}

		// For modified text
		if (segment.type != DELETE) {
			size_t len = segment.end_b - segment.start_b;
			modified_text = (char *) malloc(len + 1);
			if (modified_text) {
				strncpy(modified_text, modified + segment.start_b, len);
				modified_text[len] = '\0';
			}
		}

		// Print the diff, accounting for newlines

		if (original_text) {
			for (size_t j = 0; j < strlen(original_text); j++) {
				if (original_text[j] == '\n') {
					original_text[j] = ' ';
				}
			}
		}

		if (modified_text) {
			for (size_t j = 0; j < strlen(modified_text); j++) {
				if (modified_text[j] == '\n') {
					modified_text[j] = ' ';
				}
			}
		}

		printf(diff_fmt, segment.start_a, segment.end_a,
		       original_text ? original_text : "", diff_type, segment.start_b,
		       segment.end_b, modified_text ? modified_text : "");

		// Free allocated memory
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

	if (!old_source || !new_source || !parser) {
		return false;
	}

	// Old parse tree
	*old_tree =
	    ts_parser_parse_string(parser, NULL, old_source, strlen(old_source));
	if (!*old_tree) {
		return false;
	}

	// Textual Diff to update tree
	MSMDiffResult *result = msm_diff(old_source, new_source);
	if (!result) { // No updates
		*new_tree = *old_tree;
		return true;
	}

	for (int i = 0; i < result->count; i++) {
		MSMDiffSegment segment = result->segments[i];
		if (segment.type == EQUAL) {
			continue; // Skip equal segments
		}
		TSInputEdit edit = {
		    .start_byte = segment.start_a,
		    .old_end_byte = segment.end_a,
		    .new_end_byte = segment.end_b,
		    .start_point = {0, 0},
		    .old_end_point = {0, 0},
		    .new_end_point = {0, 0},
		};
		ts_tree_edit(*old_tree, &edit);
	}

	msm_free_diff_result(result);

	// Parse the new tree incrementally
	*new_tree = ts_parser_parse_string(parser, *old_tree, new_source,
	                                   strlen(new_source));
	if (!*new_tree) {
		ts_tree_delete(*old_tree);
		return false;
	}

	return true;
}

int main()
{

	TestSuites suites = load_test_suites("tests");
	if (suites.suites == NULL || suites.count == 0) {
		printf("Failed to load suites\n");
		return 1;
	}

	// Create a parser
	TSParser *parser = ts_parser_new();
	ts_parser_set_language(parser, tree_sitter_c());
	ts_parser_set_included_ranges(parser, NULL, 0);

	for (size_t i = 0; i < suites.count; i++) {
		TestSuite *suite = &suites.suites[i];
		printf("Test Suite %zu:\n", i + 1);
		for (size_t j = 0; j < suite->count; j++) {
			TestCase *test_case = &suite->cases[j];
			if (strcmp(test_case->language, "c") != 0) continue;

			// Print case
			printf("Testing case: %s\n", test_case->name);

			TSTree *old_tree = NULL, *new_tree = NULL;
			bool res = parse_and_update(test_case->original_source,
			                            test_case->modified_source, parser,
			                            &old_tree, &new_tree);
			if (!res) {
				printf("FAILED TO PROCESS TREES\n");
				continue;
			}

			// Get the changed ranges
			unsigned int range_count;
			TSRange *ranges =
			    ts_tree_get_changed_ranges(old_tree, new_tree, &range_count);

			print_changed_ranges(ranges, range_count,
			                     test_case->modified_source);

			// Free resources
			free(ranges);
			ts_tree_delete(old_tree);
			ts_tree_delete(new_tree);
		}
	}

	ts_parser_delete(parser);

	return 0;
}
