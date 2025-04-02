#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tree_sitter/api.h>
#include <tree_sitter/tree-sitter-c.h>

#include "meyersSequenceMatcher.h"

void print_changed_ranges(TSRange *ranges, unsigned int count, char *source)
{

    unsigned int source_length = strlen(source);
    if (count == 0)
    {
        printf("No changes detected.\n");
        return;
    }

    for (unsigned int i = 0; i < count; i++)
    {
        printf("Range %u: ", i);
        printf("Start byte: %u, ", ranges[i].start_byte);
        printf("End byte: %u, ", ranges[i].end_byte);
        printf("Start point: (%u, %u), ", ranges[i].start_point.row, ranges[i].start_point.column);
        printf("End point: (%u, %u), ", ranges[i].end_point.row, ranges[i].end_point.column);
        printf("Changed code: ");

        // Add bounds checking
        unsigned int end = ranges[i].end_byte;
        if (end > source_length)
        {
            end = source_length;
        }

        for (unsigned int j = ranges[i].start_byte; j < end; j++)
        {
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
void print_side_by_side_diff(
    const char *original,
    size_t original_length,
    const char *modified,
    size_t modified_length,
    MSMDiffSegment *segments,
    int segment_count)
{
    char *diff_fmt = "[%3d:%3d] %-40.40s %c [%3d:%3d] %-40.40s\n";

    for (int i = 0; i < segment_count; i++)
    {
        MSMDiffSegment segment = segments[i];
        char diff_type = ' ';

        // Determine diff symbol
        switch (segment.type)
        {
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
        if (segment.type != INSERT)
        {
            size_t len = segment.end_a - segment.start_a;
            original_text = (char *)malloc(len + 1);
            if (original_text)
            {
                strncpy(original_text, original + segment.start_a, len);
                original_text[len] = '\0';
            }
        }

        // For modified text
        if (segment.type != DELETE)
        {
            size_t len = segment.end_b - segment.start_b;
            modified_text = (char *)malloc(len + 1);
            if (modified_text)
            {
                strncpy(modified_text, modified + segment.start_b, len);
                modified_text[len] = '\0';
            }
        }

        // Print the diff, accounting for newlines

        if (original_text)
        {
            for (size_t j = 0; j < strlen(original_text); j++)
            {
                if (original_text[j] == '\n')
                {
                    original_text[j] = ' ';
                }
            }
        }

        if (modified_text)
        {
            for (size_t j = 0; j < strlen(modified_text); j++)
            {
                if (modified_text[j] == '\n')
                {
                    modified_text[j] = ' ';
                }
            }
        }

        printf(diff_fmt,
               segment.start_a, segment.end_a, original_text ? original_text : "",
               diff_type,
               segment.start_b, segment.end_b, modified_text ? modified_text : "");

        // Free allocated memory
        free(original_text);
        free(modified_text);
    }
}

int main()
{

    char *original_code =
        "int main() {\n"
        "    int x = 0, y = 1;\n"
        "    if (x < y) {\n"
        "        return x;\n"
        "    } else {\n"
        "        return y;\n"
        "    }\n";
    int original_length = strlen(original_code);

    char *modified_code =
        "int main() {\n"
        "    int x = 0, y = 1;\n"
        "    /* An extraordinary comment */\n"
        "    if (x > y) {\n"
        "        return x;\n"
        "    } else {\n"
        "        return y;\n"
        "    }\n";
    int modified_length = strlen(modified_code);

    // Create a parser
    TSParser *parser = ts_parser_new();
    ts_parser_set_language(parser, tree_sitter_c());
    ts_parser_set_included_ranges(parser, NULL, 0);

    // Parse the original code
    TSTree *tree = ts_parser_parse_string(parser, NULL, original_code, original_length);
    if (!tree)
    {
        fprintf(stderr, "Failed to parse the original code\n");
        return 1;
    }

    // Edit the original tree to reflect the changes (< to >)
    MSMDiffResult *result = msm_diff(original_code, modified_code);
    if (result)
    {
        for (int i = 0; i < result->count; i++)
        {
            MSMDiffSegment segment = result->segments[i];
            if (segment.type == EQUAL)
            {
                continue; // Skip equal segments
            }
            printf("Segment %d: Type %s\t| A[%d:%d], B[%d:%d]\n",
                   i,
                   segment.type == EQUAL ? "EQUAL" : segment.type == INSERT ? "INSERT"
                                                 : segment.type == DELETE   ? "DELETE"
                                                                            : "REPLACE",
                   segment.start_a,
                   segment.end_a,
                   segment.start_b,
                   segment.end_b);

            // This only works because the files are UTF-8 encoded
            TSInputEdit edit = {
                .start_byte = segment.start_a,
                .old_end_byte = segment.end_a,
                .new_end_byte = segment.end_b,
                .start_point = {0, 0},
                .old_end_point = {0, 0},
                .new_end_point = {0, 0},
            };
            ts_tree_edit(tree, &edit);
        }
        printf("\n");

        // Print side-by-side diff
        print_side_by_side_diff(original_code, original_length, modified_code, modified_length, result->segments, result->count);
        printf("\n");

        // Free the diff result
        msm_free_diff_result(result);
    }

    // Parse the modified code using the old tree
    TSTree *new_tree = ts_parser_parse_string(parser, tree, modified_code, modified_length);
    if (!new_tree)
    {
        fprintf(stderr, "Failed to parse the modified code\n");
        return 1;
    }

    // Get the changed ranges
    unsigned int range_count;
    TSRange *ranges = ts_tree_get_changed_ranges(tree, new_tree, &range_count);

    print_changed_ranges(ranges, range_count, modified_code);

    // Free resources
    free(ranges);
    ts_tree_delete(tree);
    ts_tree_delete(new_tree);
    ts_parser_delete(parser);

    return 0;
}