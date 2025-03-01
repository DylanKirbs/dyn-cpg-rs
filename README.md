# CPI7: Graph based intermediate representation for language agnostic program analysis

## Co-supervised by Cornelia Inggs and Willem Bester
> **TLDR: Develop an intermediate representation based on graph databases for storing code property graphs used in program analysis.**

Our research group develops tools and techniques for program analysis, which can be used to find violations in programs or analyse code submissions to provide feedback to students. We have developed a static analysis tool that parses source code using the incremental parser library, Tree-sitter, and builds language agnostic Code Property Graphs (CPGs) which includes data-flow and control-flow information.

Static analysis is often performed during the Continuous Integration (CI) process to generate reports of compliance issues or code quality or periodically in IDEs to highlight rule violations as the programmer is writing code. Current research in static analysis focus on the development of solutions that support the modern development style of frequent making incremental changes to source code and including many library dependencies, which result in large code bases. One of the more recent solutions is to use a graph database format for storing the CPGs to support incremental updates.

Your task would be to develop such a storage format for our tool’s CPG, similar to the database used by Joern, so that it can support incremental changes and scale well for large systems.


| Focus                     | Research and Software Engineering |
| ------------------------- | --------------------------------- |
| Broad / Deep              | Deep                              |
| Publication               | Possible                          |
| Scope for continued study | Yes                               |

Wiki available [here](https://git.cs.sun.ac.za/Computer-Science/rw771/2025/25853805-CPI7-doc/-/wikis/home)
