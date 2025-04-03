# Tests for `tree-sitter-diff`

Each test **MUST** take the format of:
```
===
<test name>
lang: <language>
Optional comments or additional information
===

<original source>

---

<modified source>

---

<expected tree-sitter output>

===
Example Case Name
...
```

1. **Name Block**:
   1. The first line is the test name, which is followed by the language of the source code.
   2. The language **MUST** be specified in the format `lang: <language>`.
   3. The next lines are optional and can be used to provide additional information about the test case
   4. The name block **MUS** be delimited by at least three equal signs above and below.
2. **Source Code Block**:
   1. The original source code **MUST** be separated from the modified source code by three dashes.
   2. The modified source code **MUST** be separated from the expected tree-sitter output by three dashes.
   3. Each source code block **MUST** be preceded and followed by an empty line.
3. **Expected Output Block**:
   1. The expected tree-sitter output **MUST** be separated from the modified source code by three dashes.
   2. The expected tree-sitter output **MUST** be preceded and followed by an empty line.
4. **General Formatting**:
   1. The test name **should** be unique and descriptive of the test case.
   2. The source code **should** be formatted according to the language's conventions.