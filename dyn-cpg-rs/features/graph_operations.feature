Feature: Complex Graph Operations
  As a code analysis tool developer
  I want to perform complex queries and transformations on the CPG
  So that I can extract meaningful insights from codebases

  Background:
    Given I have a CPG representing a medium-sized codebase

  Scenario: Finding all paths between functions
    Given I have functions with complex call relationships
    When I query for all execution paths from function A to function B
    Then I should get all possible control flow paths
    And each path should include intermediate function calls
    And the paths should respect conditional branching

  Scenario: Identifying code clone patterns
    Given I have a codebase with potential code duplicates
    When I search for structurally similar code patterns
    Then the system should identify near-identical functions
    And it should report the similarity metrics
    And it should handle minor variations like variable names

  Scenario: Data flow analysis across function boundaries
    Given I have functions that pass data between each other
    When I trace data flow from a source variable to its uses
    Then I should see all intermediate transformations
    And the analysis should cross function call boundaries
    And it should handle complex data structures

  Scenario: Performance analysis of graph operations
    Given I have a large CPG with thousands of nodes
    When I perform complex graph traversals
    Then the operations should complete within acceptable time limits
    And memory usage should remain reasonable
    And the spatial indexing should optimize range queries

  Scenario: Graph integrity after modifications
    Given I perform multiple incremental updates
    When I validate the CPG structure
    Then all edges should have valid endpoints
    And the spatial index should be consistent
    And there should be no orphaned nodes or edges