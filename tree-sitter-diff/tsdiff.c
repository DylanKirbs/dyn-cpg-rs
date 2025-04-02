#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tree_sitter/api.h>
#include <tree_sitter/tree-sitter-c.h>

#include "meyersSequenceMatcher.h"

void print_changed_ranges(TSRange *ranges, unsigned int count, char *source)
{
    for (unsigned int i = 0; i < count; i++)
    {
        printf("Range %u: ", i);
        printf("Start byte: %u, ", ranges[i].start_byte);
        printf("End byte: %u, ", ranges[i].end_byte);
        printf("Start point: (%u, %u), ", ranges[i].start_point.row, ranges[i].start_point.column);
        printf("End point: (%u, %u), ", ranges[i].end_point.row, ranges[i].end_point.column);
        printf("Changed code: ");
        for (unsigned int j = ranges[i].start_byte; j < ranges[i].end_byte; j++)
        {
            putchar(source[j]);
        }
        putchar('\n');
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