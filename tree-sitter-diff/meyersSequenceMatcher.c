#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdio.h>

#include "meyersSequenceMatcher.h"

// Internal structures for the algorithm
typedef struct
{
    int x;
    int prev_k;
} msm_DPCell;

typedef enum
{
    DIFF_EQUAL = 0,
    DIFF_INSERT = 1,
    DIFF_DELETE = 2
} msm_RawDiffType;

typedef struct
{
    msm_RawDiffType type;
    int index;
} msm_RawDiff;

// Helper functions for MSMDiffResult management
MSMDiffResult *msm_create_diff_result(int initial_capacity)
{
    MSMDiffResult *result = (MSMDiffResult *)malloc(sizeof(MSMDiffResult));
    if (!result)
        return NULL;

    result->segments = (MSMDiffSegment *)malloc(initial_capacity * sizeof(MSMDiffSegment));
    if (!result->segments)
    {
        free(result);
        return NULL;
    }

    result->count = 0;
    result->capacity = initial_capacity;
    return result;
}

void msm_free_diff_result(MSMDiffResult *result)
{
    if (!result)
        return;

    for (int i = 0; i < result->count; i++)
    {
        free(result->segments[i].value_a);
        free(result->segments[i].value_b);
    }

    free(result->segments);
    free(result);
}

bool msm_add_dff_segment(MSMDiffResult *result, MSMDiffSegment segment)
{
    if (result->count >= result->capacity)
    {
        int new_capacity = result->capacity * 2;
        MSMDiffSegment *new_segments = (MSMDiffSegment *)realloc(result->segments, new_capacity * sizeof(MSMDiffSegment));
        if (!new_segments)
            return false;

        result->segments = new_segments;
        result->capacity = new_capacity;
    }

    result->segments[result->count++] = segment;
    return true;
}

// Function to extract substring from string array
char *msm_substr_from_str_arr(char **strings, int start, int end)
{
    if (start == end)
        return strdup("");

    int total_len = 0;
    for (int i = start; i < end; i++)
    {
        total_len += strlen(strings[i]) + 1; // +1 for newline
    }

    char *result = (char *)malloc(total_len + 1);
    if (!result)
        return NULL;

    result[0] = '\0';
    for (int i = start; i < end; i++)
    {
        strcat(result, strings[i]);
        if (i < end - 1)
            strcat(result, "\n");
    }

    return result;
}

// Function to extract substring from string
char *msm_substr_from_str(const char *str, int start, int end)
{
    if (start == end)
        return strdup("");

    int len = end - start;
    char *result = (char *)malloc(len + 1);
    if (!result)
        return NULL;

    strncpy(result, str + start, len);
    result[len] = '\0';

    return result;
}

// Core Myers diff algorithm implementation
static msm_RawDiff *msm_diff_generic(
    const void *a_data, int a_len,
    const void *b_data, int b_len,
    int *diff_count,
    bool (*compare_fn)(const void *a, const void *b, int idx_a, int idx_b))
{
    int max_size = a_len + b_len;

    // Allocate DP array
    msm_DPCell *dp_array = (msm_DPCell *)malloc(sizeof(msm_DPCell) * (max_size + 1) * (2 * max_size + 1));
    if (!dp_array)
    {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    // Allocate frontier array
    int *frontier = (int *)malloc(sizeof(int) * (2 * max_size + 3));
    if (!frontier)
    {
        free(dp_array);
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    // Initialize frontier
    memset(frontier, 0, sizeof(int) * (2 * max_size + 3));
    frontier[max_size + 2] = 0;

    // Process edit script
    int final_d = -1;
    int final_k = 0;

    for (int d = 0; d <= a_len + b_len; ++d)
    {
        for (int k = -d; k <= d; k += 2)
        {
            int frontier_index = k + max_size + 1;
            bool go_down = (k == -d || (k != d && frontier[frontier_index - 1] < frontier[frontier_index + 1]));

            int x = go_down ? frontier[frontier_index + 1] : frontier[frontier_index - 1] + 1;
            int y = x - k;

            // Snake
            while (x < a_len && y < b_len && compare_fn(a_data, b_data, x, y))
            {
                ++x;
                ++y;
            }

            // Store history
            int dp_index = d * (2 * max_size + 1) + (k + max_size);
            dp_array[dp_index].x = x;
            dp_array[dp_index].prev_k = (go_down ? frontier_index + 1 : frontier_index - 1) - max_size - 1;

            if (x >= a_len && y >= b_len)
            {
                final_d = d;
                final_k = k;
                break;
            }
            else
            {
                frontier[frontier_index] = x;
            }
        }

        if (final_d != -1)
            break;
    }

    if (final_d == -1)
    {
        free(frontier);
        free(dp_array);
        fprintf(stderr, "Could not find edit script\n");
        return NULL;
    }

    // Backtrack to find edit script
    int max_raw_diff = a_len + b_len + 1;
    msm_RawDiff *raw_diffs = (msm_RawDiff *)malloc(sizeof(msm_RawDiff) * max_raw_diff);
    if (!raw_diffs)
    {
        free(frontier);
        free(dp_array);
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    int raw_diff_count = 0;
    int d = final_d;
    int k = final_k;

    while (d > 0)
    {
        int dp_index = d * (2 * max_size + 1) + (k + max_size);
        int curr_x = dp_array[dp_index].x;
        int prev_k = dp_array[dp_index].prev_k;
        int curr_y = curr_x - k;

        int prev_dp_index = (d - 1) * (2 * max_size + 1) + (prev_k + max_size);
        int prev_x = dp_array[prev_dp_index].x;
        int prev_y = prev_x - prev_k;

        // Record equal elements in snake
        while (curr_x > prev_x && curr_y > prev_y)
        {
            if (raw_diff_count < max_raw_diff)
            {
                raw_diffs[raw_diff_count].type = DIFF_EQUAL;
                raw_diffs[raw_diff_count].index = curr_x - 1;
                raw_diff_count++;
            }
            curr_x--;
            curr_y--;
        }

        // Record insert or delete
        if (curr_y > prev_y)
        {
            if (raw_diff_count < max_raw_diff)
            {
                raw_diffs[raw_diff_count].type = DIFF_INSERT;
                raw_diffs[raw_diff_count].index = curr_y - 1;
                raw_diff_count++;
            }
        }
        else if (curr_x > prev_x)
        {
            if (raw_diff_count < max_raw_diff)
            {
                raw_diffs[raw_diff_count].type = DIFF_DELETE;
                raw_diffs[raw_diff_count].index = curr_x - 1;
                raw_diff_count++;
            }
        }

        d--;
        k = prev_k;
    }

    // Process the last part (d=0)
    int curr_x = dp_array[0 * (2 * max_size + 1) + (0 + max_size)].x;
    int curr_y = curr_x;

    while (curr_x > 0)
    {
        if (raw_diff_count < max_raw_diff)
        {
            raw_diffs[raw_diff_count].type = DIFF_EQUAL;
            raw_diffs[raw_diff_count].index = curr_x - 1;
            raw_diff_count++;
        }
        curr_x--;
        curr_y--;
    }

    // Free resources we don't need anymore
    free(frontier);
    free(dp_array);

    // Reverse the raw_diffs array
    for (int i = 0; i < raw_diff_count / 2; i++)
    {
        msm_RawDiff temp = raw_diffs[i];
        raw_diffs[i] = raw_diffs[raw_diff_count - 1 - i];
        raw_diffs[raw_diff_count - 1 - i] = temp;
    }

    *diff_count = raw_diff_count;
    return raw_diffs;
}

// Comparison function for characters
static bool compare_chars(const void *a_data, const void *b_data, int idx_a, int idx_b)
{
    const char *a_str = (const char *)a_data;
    const char *b_str = (const char *)b_data;
    return a_str[idx_a] == b_str[idx_b];
}

// Core Myers diff algorithm for characters in strings
msm_RawDiff *msm_diff_chars(const char *a_str, int a_len, const char *b_str, int b_len, int *diff_count)
{
    return msm_diff_generic(a_str, a_len, b_str, b_len, diff_count, compare_chars);
}

// Process raw diffs into higher-level diff segments
static MSMDiffResult *msm_proc_raw_diffs(
    msm_RawDiff *raw_diffs, int raw_diff_count,
    bool is_character_mode,
    const void *a_data,
    const void *b_data)
{
    MSMDiffResult *result = msm_create_diff_result(16);
    if (!result)
    {
        free(raw_diffs);
        return NULL;
    }

    // Track indices in both sequences
    int a_idx = 0;
    int b_idx = 0;

    // Keep track of the current block type and its start positions
    MSMDiffType current_type = EQUAL;
    int start_a = 0;
    int start_b = 0;

    for (int i = 0; i < raw_diff_count; i++)
    {
        msm_RawDiffType type = raw_diffs[i].type;

        if (type == DIFF_EQUAL)
        {
            // If we were tracking a non-EQUAL block, add it
            if (current_type != EQUAL && a_idx > start_a && b_idx > start_b)
            {
                MSMDiffSegment segment;
                segment.type = current_type;
                segment.start_a = start_a; // old_start
                segment.end_a = a_idx;     // old_end
                segment.start_b = start_b; // new_start
                segment.end_b = b_idx;     // new_end

                if (is_character_mode)
                {
                    segment.value_a = msm_substr_from_str((const char *)a_data, start_a, a_idx);
                    segment.value_b = msm_substr_from_str((const char *)b_data, start_b, b_idx);
                }
                else
                {
                    segment.value_a = msm_substr_from_str_arr((char **)a_data, start_a, a_idx);
                    segment.value_b = msm_substr_from_str_arr((char **)b_data, start_b, b_idx);
                }

                msm_add_dff_segment(result, segment);

                // Reset tracking
                start_a = a_idx;
                start_b = b_idx;
            }
            else if (current_type != EQUAL)
            {
                // Handle INSERT or DELETE
                if (current_type == INSERT)
                {
                    MSMDiffSegment segment;
                    segment.type = INSERT;
                    segment.start_a = start_a; // old_start
                    segment.end_a = start_a;   // old_end (no deletion in old)
                    segment.start_b = start_b; // new_start
                    segment.end_b = b_idx;     // new_end
                    segment.value_a = NULL;

                    if (is_character_mode)
                    {
                        segment.value_b = msm_substr_from_str((const char *)b_data, start_b, b_idx);
                    }
                    else
                    {
                        segment.value_b = msm_substr_from_str_arr((char **)b_data, start_b, b_idx);
                    }

                    msm_add_dff_segment(result, segment);
                }
                else if (current_type == DELETE)
                {
                    MSMDiffSegment segment;
                    segment.type = DELETE;
                    segment.start_a = start_a; // old_start
                    segment.end_a = a_idx;     // old_end
                    segment.start_b = start_b; // new_start
                    segment.end_b = start_b;   // new_end (no insertion in new)

                    if (is_character_mode)
                    {
                        segment.value_a = msm_substr_from_str((const char *)a_data, start_a, a_idx);
                    }
                    else
                    {
                        segment.value_a = msm_substr_from_str_arr((char **)a_data, start_a, a_idx);
                    }

                    segment.value_b = NULL;
                    msm_add_dff_segment(result, segment);
                }

                // Reset tracking
                start_a = a_idx;
                start_b = b_idx;
            }

            current_type = EQUAL;
            a_idx++;
            b_idx++;
        }
        else if (type == DIFF_INSERT)
        {
            if (current_type == EQUAL)
            {
                // Finalize EQUAL block if needed
                if (start_a < a_idx)
                {
                    MSMDiffSegment segment;
                    segment.type = EQUAL;
                    segment.start_a = start_a; // old_start
                    segment.end_a = a_idx;     // old_end
                    segment.start_b = start_b; // new_start
                    segment.end_b = b_idx;     // new_end

                    if (is_character_mode)
                    {
                        segment.value_a = msm_substr_from_str((const char *)a_data, start_a, a_idx);
                        segment.value_b = msm_substr_from_str((const char *)b_data, start_b, b_idx);
                    }
                    else
                    {
                        segment.value_a = msm_substr_from_str_arr((char **)a_data, start_a, a_idx);
                        segment.value_b = msm_substr_from_str_arr((char **)b_data, start_b, b_idx);
                    }

                    msm_add_dff_segment(result, segment);
                }

                // Start new block
                start_a = a_idx;
                start_b = b_idx;
                current_type = INSERT;
            }
            else if (current_type == DELETE)
            {
                // This means we have both INSERT and DELETE => REPLACE
                current_type = REPLACE;
            }

            b_idx++;
        }
        else if (type == DIFF_DELETE)
        {
            if (current_type == EQUAL)
            {
                // Finalize EQUAL block if needed
                if (start_a < a_idx)
                {
                    MSMDiffSegment segment;
                    segment.type = EQUAL;
                    segment.start_a = start_a; // old_start
                    segment.end_a = a_idx;     // old_end
                    segment.start_b = start_b; // new_start
                    segment.end_b = b_idx;     // new_end

                    if (is_character_mode)
                    {
                        segment.value_a = msm_substr_from_str((const char *)a_data, start_a, a_idx);
                        segment.value_b = msm_substr_from_str((const char *)b_data, start_b, b_idx);
                    }
                    else
                    {
                        segment.value_a = msm_substr_from_str_arr((char **)a_data, start_a, a_idx);
                        segment.value_b = msm_substr_from_str_arr((char **)b_data, start_b, b_idx);
                    }

                    msm_add_dff_segment(result, segment);
                }

                // Start new block
                start_a = a_idx;
                start_b = b_idx;
                current_type = DELETE;
            }
            else if (current_type == INSERT)
            {
                // This means we have both INSERT and DELETE => REPLACE
                current_type = REPLACE;
            }

            a_idx++;
        }
    }

    // Handle the final block
    if (a_idx > start_a || b_idx > start_b)
    {
        MSMDiffSegment segment;
        segment.type = current_type;

        if (current_type == EQUAL)
        {
            segment.start_a = start_a; // old_start
            segment.end_a = a_idx;     // old_end
            segment.start_b = start_b; // new_start
            segment.end_b = b_idx;     // new_end

            if (is_character_mode)
            {
                segment.value_a = msm_substr_from_str((const char *)a_data, start_a, a_idx);
                segment.value_b = msm_substr_from_str((const char *)b_data, start_b, b_idx);
            }
            else
            {
                segment.value_a = msm_substr_from_str_arr((char **)a_data, start_a, a_idx);
                segment.value_b = msm_substr_from_str_arr((char **)b_data, start_b, b_idx);
            }
        }
        else if (current_type == INSERT)
        {
            segment.start_a = start_a; // old_start
            segment.end_a = start_a;   // old_end (no deletion in old)
            segment.start_b = start_b; // new_start
            segment.end_b = b_idx;     // new_end

            segment.value_a = NULL;

            if (is_character_mode)
            {
                segment.value_b = msm_substr_from_str((const char *)b_data, start_b, b_idx);
            }
            else
            {
                segment.value_b = msm_substr_from_str_arr((char **)b_data, start_b, b_idx);
            }
        }
        else if (current_type == DELETE)
        {
            segment.start_a = start_a; // old_start
            segment.end_a = a_idx;     // old_end
            segment.start_b = start_b; // new_start
            segment.end_b = start_b;   // new_end (no insertion in new)

            if (is_character_mode)
            {
                segment.value_a = msm_substr_from_str((const char *)a_data, start_a, a_idx);
            }
            else
            {
                segment.value_a = msm_substr_from_str_arr((char **)a_data, start_a, a_idx);
            }

            segment.value_b = NULL;
        }
        else if (current_type == REPLACE)
        {
            segment.start_a = start_a; // old_start
            segment.end_a = a_idx;     // old_end
            segment.start_b = start_b; // new_start
            segment.end_b = b_idx;     // new_end

            if (is_character_mode)
            {
                segment.value_a = msm_substr_from_str((const char *)a_data, start_a, a_idx);
                segment.value_b = msm_substr_from_str((const char *)b_data, start_b, b_idx);
            }
            else
            {
                segment.value_a = msm_substr_from_str_arr((char **)a_data, start_a, a_idx);
                segment.value_b = msm_substr_from_str_arr((char **)b_data, start_b, b_idx);
            }
        }

        msm_add_dff_segment(result, segment);
    }

    free(raw_diffs);
    return result;
}

// Split a string into an array of lines
char **split_string_into_lines(const char *str, int *line_count)
{
    if (!str || !line_count)
        return NULL;

    // Count the number of lines
    int count = 1; // At least one line
    for (const char *p = str; *p; p++)
    {
        if (*p == '\n')
            count++;
    }

    // Allocate the array
    char **lines = (char **)malloc(count * sizeof(char *));
    if (!lines)
        return NULL;

    // Copy each line
    const char *start = str;
    int i = 0;

    for (const char *p = str;; p++)
    {
        if (*p == '\n' || *p == '\0')
        {
            size_t len = p - start;
            lines[i] = (char *)malloc(len + 1);
            if (!lines[i])
            {
                // Clean up on error
                for (int j = 0; j < i; j++)
                {
                    free(lines[j]);
                }
                free(lines);
                return NULL;
            }

            // Copy the line without the newline
            strncpy(lines[i], start, len);
            lines[i][len] = '\0';
            i++;

            if (*p == '\0')
                break;
            start = p + 1;
        }
    }

    *line_count = i;
    return lines;
}

// Helper function to free lines array
void msm_free_lines(char **lines, int count)
{
    if (!lines)
        return;

    for (int i = 0; i < count; i++)
    {
        free(lines[i]);
    }

    free(lines);
}

MSMDiffResult *msm_diff(const char *a_str, const char *b_str)
{
    int a_len = strlen(a_str);
    int b_len = strlen(b_str);

    int raw_diff_count = 0;
    msm_RawDiff *raw_diffs = msm_diff_chars(a_str, a_len, b_str, b_len, &raw_diff_count);

    if (!raw_diffs)
        return NULL;

    return msm_proc_raw_diffs(raw_diffs, raw_diff_count, true, a_str, b_str);
}

// Example usage
int test_msm()
{
    const char *a_str =
        "int main() {\n"
        "    int x = 0, y = 1;\n"
        "    if (x < y) {\n"
        "        return x;\n"
        "    } else {\n"
        "        return y;\n"
        "    }\n";

    const char *b_str =
        "int main() {\n"
        "    int x = 0, y = 1;\n"
        "    if (x > y) {\n"
        "        return x;\n"
        "    } else {\n"
        "        return y;\n"
        "    }\n";

    MSMDiffResult *result = msm_diff(a_str, b_str);
    if (result)
    {
        for (int i = 0; i < result->count; i++)
        {
            printf("Segment %d: Type %s\t| A[%d:%d], B[%d:%d]\n",
                   i,
                   result->segments[i].type == EQUAL ? "EQUAL" : result->segments[i].type == INSERT ? "INSERT"
                                                             : result->segments[i].type == DELETE   ? "DELETE"
                                                                                                    : "REPLACE",
                   result->segments[i].start_a,
                   result->segments[i].end_a,
                   result->segments[i].start_b,
                   result->segments[i].end_b);
        }

        msm_free_diff_result(result);
    }

    return 0;
}