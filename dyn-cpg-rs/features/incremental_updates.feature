Feature: Incremental Code Updates
  As a developer working on a large codebase
  I want the CPG to efficiently update when I make small changes
  So that I can get fast feedback on my code modifications

  Background:
    Given I have a C source file with a simple function

  Scenario: Adding a new statement to a function
    When I add a new statement to the function body
    And I perform an incremental update
    Then the CPG should reflect the new statement
    And the control flow should be updated correctly
    And the unchanged parts should remain intact

  Scenario: Modifying a variable name
    Given the function contains variable declarations
    When I rename a variable throughout the function
    And I perform an incremental update
    Then all references to the variable should be updated
    And the data dependencies should reflect the new name
    And the rest of the CPG should remain unchanged

  Scenario: Inserting a new function
    When I add a completely new function to the file
    And I perform an incremental update
    Then the CPG should contain both functions
    And each function should maintain its own control flow
    And the translation unit should reference both functions

  Scenario: Complex edit with multiple changes
    When I make multiple edits including adding statements and renaming variables
    And I perform an incremental update
    Then all changes should be reflected correctly
    And the CPG should maintain structural integrity
    And performance should be better than full reparse