#include "test_loader.h"

#include <dirent.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define MAX_LINE_LENGTH  4096
#define DELIMITER_LENGTH 3
#define NAME_DELIMITER   "==="
#define SOURCE_DELIMITER "---"
#define PATH_MAX         4096

#define is_line_delimited(line, delimiter)                                     \
	(strncmp((line), (delimiter), strlen(delimiter)) == 0)

/**
 * Reads the entire content of a file into a string.
 */
static char *read_file(const char *filename)
{
	size_t bytes_read, file_size;
	char *content;

	FILE *file = fopen(filename, "r");
	if (!file) {
		return NULL;
	}

	/* Get file size */
	fseek(file, 0, SEEK_END);
	file_size = ftell(file);
	rewind(file);

	/* Allocate memory for the file content plus null terminator */
	content = (char *) malloc(file_size + 1);
	if (!content) {
		fclose(file);
		return NULL;
	}

	/* Read file content */
	bytes_read = fread(content, 1, file_size, file);
	fclose(file);

	if (bytes_read != file_size) {
		free(content);
		return NULL;
	}

	content[file_size] = '\0';
	return content;
}

/**
 * Extract language from "lang: <language>" line.
 */
static char *extract_language(const char *line)
{
	const char *prefix = "lang: ";
	char *language;
	size_t len;

	if (strncmp(line, prefix, strlen(prefix)) != 0) {
		return NULL;
	}

	language = strdup(line + strlen(prefix));

	/* Remove any trailing whitespace or newline */
	len = strlen(language);
	while (len > 0 && (language[len - 1] == '\n' || language[len - 1] == '\r' ||
	                   language[len - 1] == ' ')) {
		language[--len] = '\0';
	}

	return language;
}

/**
 * Parse name block
 */
static char *parse_name_block(TestCase *test_case)
{
	/* Get the test name (next line) */
	char *line = strtok(NULL, "\n");
	if (!line) {
		return NULL;
	}

	/* Store test name */
	test_case->name = strdup(line);
	if (!test_case->name) {
		return NULL;
	}

	/* Get language line */
	line = strtok(NULL, "\n");
	if (!line) {
		free(test_case->name);
		test_case->name = NULL;
		return NULL;
	}

	/* Extract language */
	test_case->language = extract_language(line);
	if (!test_case->language) {
		free(test_case->name);
		test_case->name = NULL;
		return NULL;
	}

	/* Skip any additional comment lines until closing delimiter */
	while ((line = strtok(NULL, "\n")) != NULL) {
		if (is_line_delimited(line, NAME_DELIMITER)) {
			return line;
		}
	}

	/* Failed to find closing delimiter */
	free(test_case->name);
	test_case->name = NULL;
	free(test_case->language);
	test_case->language = NULL;
	return NULL;
}

/*
 * Parse source block
 * Returns a newly allocated string containing the content between delimiters.
 * Caller is responsible for freeing the returned memory.
 * Returns NULL on failure.
 */
static char *parse_source_block(char **source)
{
	char *buffer, *new_buffer, *line;
	size_t capacity = 256, used = 0, line_len, new_size;

	/* Null out source */
	*source = NULL;

	/* Create buffer */
	if (!(buffer = malloc(capacity))) {
		return NULL; /* Memory allocation failed */
	}
	buffer[0] = '\0';

	/* consume first delim, then collect content until we hit the final delim */
	while ((line = strtok(NULL, "\n")) != NULL) {
		if (is_line_delimited(line, NAME_DELIMITER) ||
		    is_line_delimited(line, SOURCE_DELIMITER)) {
			/* We've reached the end delimiter */
			*source = buffer;
			return line;
		}

		/* Calculate length needed for this line plus newline */
		line_len = strlen(line);
		new_size = used + line_len + 2; /* +2 for newline and null terminator */

		/* Double the buffer if needed */
		if (new_size > capacity) {
			while (capacity < new_size) {
				capacity *= 2;
			}

			new_buffer = realloc(buffer, capacity);
			if (!new_buffer) {
				free(buffer);
				return NULL; /* Memory allocation failed */
			}
			buffer = new_buffer;
		}

		/* Append line and newline to buffer */
		strcpy(buffer + used, line);
		used += line_len;
		buffer[used++] = '\n';
		buffer[used] = '\0';
	}

	/* If we get here, we've reached the end */
	*source = buffer;
	return NULL;
}

/**
 * Parse a single test file into a TestSuite.
 */
static TestSuite parse_test_file(const char *filepath)
{
	/* Tests */
	size_t current_test = 0, test_count = 0;
	TestSuite suite = {NULL, 0};
	/* Content */
	char *current_pos, *line, *content = read_file(filepath);
	if (!content) {
		return suite;
	}

	/* Count test cases and allocate memory */
	current_pos = content;
	while ((current_pos = strstr(current_pos, NAME_DELIMITER)) != NULL) {
		test_count++;
		current_pos += DELIMITER_LENGTH; /* Move past the delimiter */
	}
	test_count /= 2; /* Each test is delimited by two NAME_DELIMITER lines */
	if (test_count == 0) {
		free(content);
		return suite;
	}
	/* Allocate memory for test cases */
	suite.cases = (TestCase *) calloc(test_count, sizeof(TestCase));
	if (!suite.cases) {
		free(content);
		return suite;
	}

	/* Parse test cases */
	line = strtok(content, "\n");
	while (line && current_test < test_count) {
		if (!is_line_delimited(line, NAME_DELIMITER)) {
			line = strtok(NULL, "\n");
			continue;
		}

		if (!(line = parse_name_block(&suite.cases[current_test]))) break;

		if (!is_line_delimited(line, NAME_DELIMITER)) break;

		if (!(line = parse_source_block(
		          &suite.cases[current_test].original_source)))
			break;

		if (!is_line_delimited(line, SOURCE_DELIMITER)) break;

		if (!(line = parse_source_block(
		          &suite.cases[current_test].modified_source)))
			break;

		if (!is_line_delimited(line, SOURCE_DELIMITER)) break;

		if (!(line = parse_source_block(
		          &suite.cases[current_test].expected_output)))
			break;

		current_test++;
	}

	suite.count = current_test + 1;
	free(content);
	return suite;
}

/**
 * Load each .txt file in the directory into its own TestSuite.
 */
TestSuites load_test_suites(const char *directory)
{
	/* File */
	char filepath[PATH_MAX];
	DIR *dir = opendir(directory);
	struct dirent *entry;
	size_t txt_count = 0;
	/* Suites */
	TestSuites suites = {NULL, 0};
	size_t suite_index = 0;
	TestSuite suite;

	if (!dir) {
		return suites;
	}

	/* First pass: count .txt files */
	while ((entry = readdir(dir)) != NULL) {
		if (entry->d_type == DT_REG) { /* Regular file */
			const char *ext = strrchr(entry->d_name, '.');
			if (ext && strcmp(ext, ".txt") == 0) {
				txt_count++;
			}
		}
	}

	rewinddir(dir);

	if (txt_count == 0) {
		closedir(dir);
		return suites;
	}

	/* Allocate memory for test suites */
	suites.suites = (TestSuite *) calloc(txt_count, sizeof(TestSuite));
	if (!suites.suites) {
		closedir(dir);
		return suites;
	}

	/* Second pass: load test suites */
	while ((entry = readdir(dir)) != NULL && suite_index < txt_count) {
		if (entry->d_type == DT_REG) { /* Regular file */
			const char *ext = strrchr(entry->d_name, '.');
			if (ext && strcmp(ext, ".txt") == 0) {
				/* Construct the full path */
				snprintf(filepath, PATH_MAX, "%s/%s", directory, entry->d_name);

				/* Parse the test file */
				suite = parse_test_file(filepath);
				if (suite.cases && suite.count > 0) {
					suites.suites[suite_index] = suite;
					suite_index++;
				} else {
					/* If parsing failed, free the allocated memory */
					if (suite.cases) {
						free(suite.cases);
					}
				}
			}
		}
	}

	suites.count = suite_index;
	closedir(dir);
	return suites;
}

/**
 * Free the allocated memory for the TestSuites and its test cases.
 */
void free_test_suites(TestSuites *suites)
{
	size_t i, j;
	TestSuite *suite;
	TestCase *test_case;

	if (!suites || !suites->suites) {
		return;
	}

	for (i = 0; i < suites->count; i++) {
		suite = &suites->suites[i];

		for (j = 0; j < suite->count; j++) {
			test_case = &suite->cases[j];
			free(test_case->name);
			free(test_case->language);
			free(test_case->original_source);
			free(test_case->modified_source);
			free(test_case->expected_output);
		}

		free(suite->cases);
	}

	free(suites->suites);
	suites->suites = NULL;
	suites->count = 0;
}

/* Function to print a test case */
void print_test_case(const TestCase *test_case)
{
	if (!test_case) return;

	printf("Test Name: %s\n", test_case->name ? test_case->name : "(null)");
	printf("Language: %s\n",
	       test_case->language ? test_case->language : "(null)");
	printf("Original Source:\n%s\n",
	       test_case->original_source ? test_case->original_source : "(null)");
	printf("Modified Source:\n%s\n",
	       test_case->modified_source ? test_case->modified_source : "(null)");
	printf("Expected Output:\n%s\n",
	       test_case->expected_output ? test_case->expected_output : "(null)");
	printf("----------------------------------------\n");
}

/* Example usage */
int test_loader()
{
	size_t i, j;

	TestSuites suites = load_test_suites("tests");
	if (suites.suites == NULL || suites.count == 0) {
		fprintf(stderr, "Failed to load test suites.\n");
		return 1;
	}

	for (i = 0; i < suites.count; i++) {
		TestSuite *suite = &suites.suites[i];
		printf("Test Suite %lu:\n", i + 1);
		for (j = 0; j < suite->count; j++) {
			print_test_case(&suite->cases[j]);
		}
	}

	free_test_suites(&suites);
	return 0;
}
