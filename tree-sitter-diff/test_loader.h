#ifndef TEST_LOADER_H
#define TEST_LOADER_H

#include <stddef.h>

/**
 * Single Test Case
 */
typedef struct
{
    char *name;
    char *language;
    char *original_source;
    char *modified_source;
    char *expected_output;
} TestCase;

/**
 * Test Suite
 */
typedef struct
{
    char *name;
    TestCase *cases;
    size_t count;
} TestSuite;

/**
 * Test Suites
 */
typedef struct
{
    TestSuite *suites;
    size_t count;
} TestSuites;

/**
 * Load each .txt file in the directory into it's own TestSuite.
 * Each file is it's own suite, containing multiple test cases.
 */
TestSuites load_test_suites(const char *directory);

/**
 * Free the allocated memory for the TestSuite and its test cases.
 */
void free_test_suites(TestSuites *suites);

#endif /* TEST_LOADER_H */
