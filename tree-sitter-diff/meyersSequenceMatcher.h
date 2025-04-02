#ifndef MYERS_DIFF_H
#define MYERS_DIFF_H

#include <stdbool.h>

/**
 * Enumeration of diff types for segments
 */
typedef enum
{
    EQUAL,  /* Text segments are equal */
    INSERT, /* Text was inserted in B */
    DELETE, /* Text was deleted from A */
    REPLACE /* Text was replaced (combination of delete and insert) */
} MSMDiffType;

/**
 * Structure representing a segment of differences between two texts
 */
typedef struct
{
    MSMDiffType type; /* Type of difference */
    int start_a;      /* Start position in string A (start_b if INSERT) */
    int end_a;        /* End position in string A (start_b if INSERT) */
    int start_b;      /* Start position in string B (start_a if DELETE) */
    int end_b;        /* End position in string B (start_a if DELETE) */
    char *value_a;    /* Substring from A (NULL if INSERT) */
    char *value_b;    /* Substring from B (NULL if DELETE) */
} MSMDiffSegment;

/**
 * Structure containing the result of a diff operation
 */
typedef struct
{
    MSMDiffSegment *segments; /* Array of difference segments */
    int count;                /* Number of segments */
    int capacity;             /* Capacity of segments array */
} MSMDiffResult;

/**
 * Frees all memory associated with a diff result
 * @param result Pointer to the MSMDiffResult to free
 */
void msm_free_diff_result(MSMDiffResult *result);

/**
 * Performs character-by-character diff between two strings
 * @param a_str First string
 * @param b_str Second string
 * @return MSMDiffResult containing the differences between the strings
 */
MSMDiffResult *msm_diff(const char *a_str, const char *b_str);

#endif /* MYERS_DIFF_H */