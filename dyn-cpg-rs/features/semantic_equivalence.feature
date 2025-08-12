Feature: Cross-Language Semantic Equivalence
  As a developer working with multi-language codebases
  I want to compare code semantics across different programming languages
  So that I can ensure consistent behavior in polyglot systems

  Background:
    Given I have equivalent algorithms implemented in different languages

  Scenario: Comparing equivalent C and Python functions
    Given I have a simple sorting function in C
    And I have the same sorting logic in Python
    When I generate CPGs for both implementations
    Then the CPGs should be semantically equivalent
    And the control flow patterns should match
    And the data dependencies should be structurally similar

  Scenario: Detecting semantic differences in similar code
    Given I have two C functions that look similar
    And one has a subtle logic error
    When I compare their CPGs
    Then the comparison should identify the structural differences
    And it should highlight the specific divergent paths
    And the report should be human-readable

  Scenario: Language-specific construct handling
    Given I have code using language-specific constructs
    And Python list comprehensions and C pointer arithmetic
    When I generate CPGs for both
    Then each should preserve the language semantics
    And cross-language comparisons should focus on algorithmic structure
    And language differences should not mask logical equivalence

  Scenario: Function signature evolution
    Given I have the same function across multiple versions
    When the function signature changes but logic remains the same
    Then the CPGs should identify semantic preservation
    And structural changes should be clearly separated from logical changes