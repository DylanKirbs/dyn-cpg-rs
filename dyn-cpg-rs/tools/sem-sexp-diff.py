#!/usr/bin/env python3
from curses.ascii import SP
from platform import node
import sys
import os
import re
import argparse
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
from difflib import SequenceMatcher

# Configuration
SEMANTIC_MISMATCH_COST = 100
STRUCTURAL_MISMATCH_COST = 10
SPAN_DISTANCE_WEIGHT = 1
SPAN_THRESHOLD = 25


@dataclass
class Node:
    node_type: str
    properties: Dict[str, str] = field(default_factory=dict)
    original_id: Optional[str] = None
    canonical_id: Optional[str] = None
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)

    def get_kind(self) -> str:
        """Get the most important identifier - kind or node type"""
        return self.properties.get("kind", self.node_type)

    def semantic_signature(self) -> Tuple:
        """Create a signature for semantic matching"""
        # Sort properties for consistent hashing, exclude id fields
        props = {
            k: v
            for k, v in self.properties.items()
            if k not in {"id", "canonical-id", "span"}
        }
        return (self.get_kind(), tuple(sorted(props.items())))

    def structural_path(self, nodes: Dict[str, "Node"]) -> List[str]:
        """Get the path from this node to the root as a list of node types/kinds"""
        path = []
        current = self
        visited = set()

        while current:
            node_id = current.original_id
            if node_id in visited:
                break  # cycle detected
            visited.add(node_id)
            path.append(current.get_kind())
            current = nodes.get(current.parent_id or "", None)

        return list(reversed(path))

    def pretty(self) -> str:
        """Pretty print the node for diff output"""
        props = ", ".join(f"{k}='{v}'" for k, v in self.properties.items())
        return f"{self.node_type}({props})(edges={len(self.children)})"


@dataclass
class Edge:
    edge_type: str
    source: str  # source node id
    target: str  # target node id

    def pretty(self) -> str:
        """Pretty print the edge for diff output"""
        return f"{self.source} -> [{self.edge_type}] {self.target}"


class SimpleSExpParser:
    """Simplified S-expression parser for CPG files"""

    def __init__(self, content: str):
        self.content = content
        self.nodes = {}  # id -> Node
        self.edges = []  # List of Edge objects
        self.line_map = {}  # node_id -> line number

    def _build_tree_structure(self):
        """Build parent-child relationships from the parsed nodes and edges"""
        # Look for SyntaxChild edges to build tree structure
        for edge in self.edges:
            if (
                edge.edge_type == "SyntaxChild"
                and edge.target in self.nodes
                and edge.source in self.nodes
            ):
                # Source is parent, target is child
                parent_node = self.nodes[edge.source]
                child_node = self.nodes[edge.target]

                if parent_node is child_node:
                    continue

                if edge.target not in parent_node.children:
                    parent_node.children.append(edge.target)

                child_node.parent_id = edge.source

    def parse(self) -> Tuple[Dict[str, Node], List[Edge], Dict[str, int]]:
        """Parse S-expression using regex patterns"""
        lines = self.content.split("\n")

        # Pattern to match node definitions with their properties
        # Matches: (NodeType :prop1 "value1" :prop2 "value2" ...)
        node_pattern = r'\((\w+(?:\([^)]*\))?)\s*((?::[^:\s]+\s+"[^"]*"\s*)*)'

        # Pattern to match edges
        # Matches: (-> EdgeType (TargetNode ...))
        edge_pattern = r"\(->\s+(\w+)\s+\("

        # Pattern to match visited references
        # Matches: [visited "id"]
        visited_pattern = r'\[visited\s+"([^"]+)"\]'

        current_node_id = None

        for line_num, line in enumerate(lines, 1):
            # Extract nodes with properties
            for match in re.finditer(node_pattern, line):
                node_type = match.group(1)
                props_str = match.group(2)

                # Parse properties
                properties = {}
                prop_matches = re.findall(r':([^:\s]+)\s+"([^"]*)"', props_str)
                for prop_name, prop_value in prop_matches:
                    properties[prop_name] = prop_value

                # Create node
                node = Node(
                    node_type=node_type,
                    properties=properties,
                    original_id=properties.get("id"),
                    canonical_id=properties.get("canonical-id"),
                )

                if node.original_id:
                    self.nodes[node.original_id] = node
                    self.line_map[node.original_id] = line_num
                    current_node_id = node.original_id

            # Extract edges
            if current_node_id:
                # Look for edge definitions
                # Look for edge definitions: (-> EdgeType (NodeType ...))
                edge_match = re.search(
                    r"\(->\s+(\w+)\s+\(([^)]+\([^)]*\)[^)]*|[^)]+)\)", line
                )
                if edge_match:
                    edge_type = edge_match.group(1)
                    node_content = edge_match.group(2)

                    # Extract node type and properties from node_content
                    node_match = re.match(
                        r'(\w+(?:\([^)]*\))?)\s*((?::[^:\s]+\s+"[^"]*"\s*)*)',
                        node_content,
                    )
                    if node_match:
                        node_type = node_match.group(1)
                        props_str = node_match.group(2)

                        # Parse properties
                        properties = {}
                        prop_matches = re.findall(r':([^:\s]+)\s+"([^"]*)"', props_str)
                        for prop_name, prop_value in prop_matches:
                            properties[prop_name] = prop_value

                        # Create target node
                        target_id = properties.get("id")
                        if not target_id:
                            target_id = properties.get("canonical-id")

                        if target_id:
                            # Create or update the node
                            if target_id not in self.nodes:
                                target_node = Node(
                                    node_type=node_type,
                                    properties=properties,
                                    original_id=properties.get("id"),
                                    canonical_id=properties.get("canonical-id"),
                                )
                                self.nodes[target_id] = target_node
                                self.line_map[target_id] = line_num

                            # Create edge
                            edge = Edge(edge_type, current_node_id, target_id)
                            self.edges.append(edge)

                # Look for visited references
                for visited_match in re.finditer(visited_pattern, line):
                    target_id = visited_match.group(1)
                    # This creates an implicit edge - we could determine the type from context
                    # For now, we'll call it a "Reference" edge
                    edge = Edge("Reference", current_node_id, target_id)
                    self.edges.append(edge)

        # Build tree structure from the parsed edges
        self._build_tree_structure()

        return self.nodes, self.edges, self.line_map


def parse_span(span_str: Optional[str]) -> Optional[Tuple[int, int]]:
    """Parse a span string like '(123, 456)' into (start, end) tuple"""
    if not span_str:
        return None
    match = re.match(r"\((\d+),\s*(\d+)\)", span_str)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None


def span_distance(span1: Optional[str], span2: Optional[str]) -> int:
    s1 = parse_span(span1)
    s2 = parse_span(span2)
    if s1 is None or s2 is None:
        return SPAN_THRESHOLD + 1

    start1, end1 = s1
    start2, end2 = s2
    return abs(start1 - start2) + abs(end1 - end2)


def path_similarity_score(path1: List[str], path2: List[str]) -> float:
    if not path1 and not path2:
        return 1.0
    if not path1 or not path2:
        return 0.0
    matcher = SequenceMatcher(None, path1, path2)
    return matcher.ratio()


def calculate_edge_similarity(
    edge1: Edge,
    edge2: Edge,
    node_matches: Dict[str, str],
    old_nodes: Dict[str, Node],
    new_nodes: Dict[str, Node],
) -> float:
    """Score similarity between two edges based on matched endpoints, type, and semantic context."""
    src1, tgt1 = edge1.source, edge1.target
    src2, tgt2 = edge2.source, edge2.target

    # Check if sources and targets are matched
    src_matched = node_matches.get(src1) == src2
    tgt_matched = node_matches.get(tgt1) == tgt2

    # If neither endpoint matches, this is likely not the same edge
    if not src_matched and not tgt_matched:
        return 0.0

    score = 0.0

    # Strong bonus for exact endpoint matches
    if src_matched and tgt_matched:
        score += 90.0  # Very strong match if both ends align perfectly
    elif src_matched or tgt_matched:
        # Check if the unmatched endpoint has the same semantic signature
        if src_matched and tgt1 in old_nodes and tgt2 in new_nodes:
            old_tgt = old_nodes[tgt1]
            new_tgt = new_nodes[tgt2]
            if old_tgt.semantic_signature() == new_tgt.semantic_signature():
                score += 60.0  # Good match with semantic equivalence
            else:
                score += 25.0  # Weak match
        elif tgt_matched and src1 in old_nodes and src2 in new_nodes:
            old_src = old_nodes[src1]
            new_src = new_nodes[src2]
            if old_src.semantic_signature() == new_src.semantic_signature():
                score += 60.0  # Good match with semantic equivalence
            else:
                score += 25.0  # Weak match
        else:
            score += 20.0  # Basic partial match

    # Edge type bonus
    if edge1.edge_type == edge2.edge_type:
        score += 10.0  # Bonus for matching edge type
    else:
        # Some edge types are semantically similar
        similar_types = [
            {"SyntaxChild", "SyntaxSibling"},
            {"ControlFlowTrue", "ControlFlowFalse", "ControlFlowEpsilon"},
        ]
        for similar_group in similar_types:
            if edge1.edge_type in similar_group and edge2.edge_type in similar_group:
                score += 5.0  # Small bonus for similar edge types
                break

    return min(100.0, score)


def fuzzy_match_edges(
    old_edges: List[Edge],
    new_edges: List[Edge],
    node_matches: Dict[str, str],
    old_nodes: Dict[str, Node],
    new_nodes: Dict[str, Node],
) -> Dict[int, int]:  # old_edge_index -> new_edge_index
    """Match edges based on fuzzy node matches and edge type."""
    matches = {}
    used_new = set()

    # Score all pairs
    scored_pairs = []
    for i, old_edge in enumerate(old_edges):
        for j, new_edge in enumerate(new_edges):
            score = calculate_edge_similarity(
                old_edge, new_edge, node_matches, old_nodes, new_nodes
            )
            if score > 30.0:  # Lower threshold but with better scoring
                scored_pairs.append((score, i, j))

    # Greedy match best first
    scored_pairs.sort(reverse=True)
    for score, i, j in scored_pairs:
        if j not in used_new:
            matches[i] = j
            used_new.add(j)

    return matches


def fuzzy_edge_diff(
    old_edges: List[Edge],
    new_edges: List[Edge],
    old_nodes: Dict[str, Node],
    new_nodes: Dict[str, Node],
) -> List[str]:

    node_matches = fuzzy_match_nodes(old_nodes, new_nodes)
    edge_matches = fuzzy_match_edges(
        old_edges, new_edges, node_matches, old_nodes, new_nodes
    )

    edge_changes = []
    matched_old_indices = set(edge_matches.keys())
    matched_new_indices = set(edge_matches.values())

    # Focus on meaningful edge types (filter out noisy Reference edges)
    important_edge_types = {
        "SyntaxChild",
        "SyntaxSibling",
        "ControlFlowTrue",
        "ControlFlowFalse",
        "ControlFlowEpsilon",
        "ControlFlowFunctionReturn",
        "PDData",
    }

    # Changed edges (matched but different type or endpoints remapped)
    for old_idx, new_idx in edge_matches.items():
        old_edge = old_edges[old_idx]
        new_edge = new_edges[new_idx]

        # Skip noisy Reference edges unless they represent significant changes
        if old_edge.edge_type == "Reference" and new_edge.edge_type == "Reference":
            continue

        old_src_name = (
            old_nodes[old_edge.source].get_kind()
            if old_edge.source in old_nodes
            else "unknown"
        )
        old_tgt_name = (
            old_nodes[old_edge.target].get_kind()
            if old_edge.target in old_nodes
            else "unknown"
        )
        new_src_name = (
            new_nodes[new_edge.source].get_kind()
            if new_edge.source in new_nodes
            else "unknown"
        )
        new_tgt_name = (
            new_nodes[new_edge.target].get_kind()
            if new_edge.target in new_nodes
            else "unknown"
        )

        # Only report if there's a meaningful change in important edge types
        has_meaningful_change = (
            old_edge.edge_type != new_edge.edge_type
            or node_matches.get(old_edge.source) != new_edge.source
            or node_matches.get(old_edge.target) != new_edge.target
        )

        if has_meaningful_change and (
            old_edge.edge_type in important_edge_types
            or new_edge.edge_type in important_edge_types
        ):
            edge_changes.append(
                f"~ Modified Edge: [{old_edge.edge_type}] {old_src_name} -> {old_tgt_name} "
                f"→ [{new_edge.edge_type}] {new_src_name} -> {new_tgt_name}"
            )

    # Deleted edges - only report important ones
    for i, edge in enumerate(old_edges):
        if i not in matched_old_indices and edge.edge_type in important_edge_types:
            src_name = (
                old_nodes[edge.source].get_kind()
                if edge.source in old_nodes
                else "unknown"
            )
            tgt_name = (
                old_nodes[edge.target].get_kind()
                if edge.target in old_nodes
                else "unknown"
            )
            edge_changes.append(
                f"- Deleted Edge: [{edge.edge_type}] {src_name} -> {tgt_name}"
            )

    # Added edges - only report important ones
    for j, edge in enumerate(new_edges):
        if j not in matched_new_indices and edge.edge_type in important_edge_types:
            src_name = (
                new_nodes[edge.source].get_kind()
                if edge.source in new_nodes
                else "unknown"
            )
            tgt_name = (
                new_nodes[edge.target].get_kind()
                if edge.target in new_nodes
                else "unknown"
            )
            edge_changes.append(
                f"+ Added Edge: [{edge.edge_type}] {src_name} -> {tgt_name}"
            )

    # If no meaningful edge changes, provide summary
    if not edge_changes:
        total_old_important = sum(
            1 for e in old_edges if e.edge_type in important_edge_types
        )
        total_new_important = sum(
            1 for e in new_edges if e.edge_type in important_edge_types
        )
        if total_old_important != total_new_important:
            edge_changes.append(
                f"Edge count changed: {total_old_important} -> {total_new_important} important edges"
            )
        else:
            edge_changes.append("No significant edge changes detected.")

    return edge_changes


def calculate_node_similarity(
    node1: Node, node2: Node, nodes1: Dict[str, Node], nodes2: Dict[str, Node]
) -> float:
    """Calculate similarity score between two nodes (higher = more similar)"""

    # Primary criteria: node type/kind must match exactly
    if node1.get_kind() != node2.get_kind():
        return 0.0  # No match if types don't match

    similarity_score = 100.0  # Start with perfect match

    # Secondary criteria: structural path similarity
    path1 = node1.structural_path(nodes1)
    path2 = node2.structural_path(nodes2)

    path_sim = path_similarity_score(path1, path2)
    similarity_score += path_sim * 30.0 - 15.0

    # Tertiary criteria: span distance (smaller distance = higher similarity)
    span1 = node1.properties.get("span")
    span2 = node2.properties.get("span")

    if span1 and span2:
        distance = span_distance(span1, span2)
        if distance <= SPAN_THRESHOLD:
            # Give bonus for close spans
            span_bonus = (SPAN_THRESHOLD - distance) / SPAN_THRESHOLD * 10.0
            similarity_score += span_bonus
        else:
            # Penalty for far spans
            similarity_score -= min(20.0, distance / SPAN_DISTANCE_WEIGHT)

    # Additional semantic properties match (fine-tuning)
    props1 = {
        k: v
        for k, v in node1.properties.items()
        if k not in {"id", "canonical-id", "span"}
    }
    props2 = {
        k: v
        for k, v in node2.properties.items()
        if k not in {"id", "canonical-id", "span"}
    }

    if props1 == props2:
        similarity_score += 5.0  # Bonus for exact property match
    else:
        # Calculate partial property match
        all_props = set(props1.keys()) | set(props2.keys())
        if all_props:
            matching_props = sum(
                1 for prop in all_props if props1.get(prop) == props2.get(prop)
            )
            prop_ratio = matching_props / len(all_props)
            similarity_score += prop_ratio * 5.0 - 2.5  # -2.5 to +2.5 range

    return max(0.0, similarity_score)


def fuzzy_match_nodes(
    old_nodes: Dict[str, Node], new_nodes: Dict[str, Node]
) -> Dict[str, str]:
    """
    Perform fuzzy matching between old and new nodes.
    Returns mapping from old_node_id -> new_node_id for matched pairs.
    """
    matches = {}
    old_ids = list(old_nodes.keys())
    new_ids = list(new_nodes.keys())

    # Build similarity matrix
    similarity_matrix = {}

    for old_id in old_ids:
        old_node = old_nodes[old_id]
        best_match = None
        best_score = 0.0

        for new_id in new_ids:
            new_node = new_nodes[new_id]
            score = calculate_node_similarity(old_node, new_node, old_nodes, new_nodes)

            if score > best_score and score > 50.0:  # Minimum threshold for a match
                best_score = score
                best_match = new_id

        if best_match:
            matches[old_id] = best_match

    # Ensure one-to-one mapping (greedy approach)
    # If multiple old nodes match to the same new node, keep the best match
    final_matches = {}
    used_new_ids = set()

    # Sort by similarity score (descending)
    candidate_matches = []
    for old_id, new_id in matches.items():
        old_node = old_nodes[old_id]
        new_node = new_nodes[new_id]
        score = calculate_node_similarity(old_node, new_node, old_nodes, new_nodes)
        candidate_matches.append((score, old_id, new_id))

    candidate_matches.sort(reverse=True)  # Best matches first

    for score, old_id, new_id in candidate_matches:
        if new_id not in used_new_ids:
            final_matches[old_id] = new_id
            used_new_ids.add(new_id)

    return final_matches


def fuzzy_node_diff(
    old_nodes: Dict[str, Node], new_nodes: Dict[str, Node]
) -> List[str]:
    """Enhanced diff using fuzzy matching"""

    # Perform fuzzy matching
    matches = fuzzy_match_nodes(old_nodes, new_nodes)

    # Collect all changes with their span positions
    node_changes = []

    # Find nodes that changed (matched but with different properties)
    for old_id, new_id in matches.items():
        old_node = old_nodes[old_id]
        new_node = new_nodes[new_id]

        # Check if properties changed (excluding IDs)
        old_props = {
            k: v
            for k, v in old_node.properties.items()
            if k not in {"id", "canonical-id"}
        }
        new_props = {
            k: v
            for k, v in new_node.properties.items()
            if k not in {"id", "canonical-id"}
        }

        span_result = parse_span(old_node.properties.get("span", ""))
        span_start = span_result[0] if span_result else 0
        if (old_props != new_props) or (
            len(old_node.children) != len(new_node.children)
        ):

            change_lines = [
                f"~ Modified: {old_node.node_type} (old: {old_id}, new: {new_id})"
            ]

            # Show specific property changes
            all_props = set(old_props.keys()) | set(new_props.keys())
            for prop in sorted(all_props):
                old_val = old_props.get(prop)
                new_val = new_props.get(prop)
                if old_val != new_val:
                    if old_val is None:
                        change_lines.append(f"  +{prop}: '{new_val}'")
                    elif new_val is None:
                        change_lines.append(f"  -{prop}: '{old_val}'")
                    else:
                        change_lines.append(f"  ~{prop}: '{old_val}' -> '{new_val}'")

            # Number of children changed
            if len(old_node.children) != len(new_node.children):
                change_lines.append(
                    f"  ~ children: {len(old_node.children)} -> {len(new_node.children)}"
                )

        # else:
        #     node_changes.append(
        #         (
        #             span_start,
        #             "unchanged",
        #             [f" {old_node.pretty()} | {new_node.pretty()}"],
        #         )
        #     )

    # Find nodes that are only in old (deleted) - those not matched
    matched_old_ids = set(matches.keys())
    for old_id, node in old_nodes.items():
        if old_id not in matched_old_ids:
            span_result = parse_span(node.properties.get("span", ""))
            span_start = span_result[0] if span_result else 0
            node_changes.append(
                (span_start, "deleted", [f"- Deleted: {node.pretty()}"])
            )

    # Find nodes that are only in new (added) - those not matched
    matched_new_ids = set(matches.values())
    for new_id, node in new_nodes.items():
        if new_id not in matched_new_ids:
            span_result = parse_span(node.properties.get("span", ""))
            span_start = span_result[0] if span_result else 0
            node_changes.append((span_start, "added", [f"+ Added: {node.pretty()}"]))

    # Sort all changes by span position, then by type priority
    type_priority = {"deleted": 0, "added": 1, "modified": 2, "unchanged": 3}
    node_changes.sort(key=lambda x: (x[0], type_priority[x[1]]))

    # Flatten the output
    diff_lines = []
    for _, change_type, lines in node_changes:
        diff_lines.extend(lines)

    return diff_lines


def fuzzy_graph_diff(
    old_nodes: Dict[str, Node],
    new_nodes: Dict[str, Node],
    old_edges: List[Edge],
    new_edges: List[Edge],
) -> List[str]:
    diff = []

    node_diff_lines = fuzzy_node_diff(old_nodes, new_nodes)

    if node_diff_lines:
        diff.append("\n~~~ Node Changes ~~~\n")
        diff.extend(node_diff_lines)
    else:
        diff.append("\n~~~ No Node Changes ~~~\n")

    edge_diff_lines = fuzzy_edge_diff(old_edges, new_edges, old_nodes, new_nodes)

    if edge_diff_lines:
        diff.append("\n~~~ Edge Changes ~~~\n")
        diff.extend(edge_diff_lines)
    else:
        diff.append("\n~~~ No Edge Changes ~~~\n")

    return diff


def main():
    global SPAN_THRESHOLD, SEMANTIC_MISMATCH_COST, STRUCTURAL_MISMATCH_COST, SPAN_DISTANCE_WEIGHT

    parser = argparse.ArgumentParser(description="Simple CPG S-expression diff tool")
    parser.add_argument(
        "files", nargs=2, metavar=("FILE1", "FILE2"), help="Files to compare"
    )
    parser.add_argument(
        "--span-threshold",
        type=int,
        default=SPAN_THRESHOLD,
        help="Max span distance to consider similar (default: %(default)s)",
    )
    parser.add_argument(
        "--semantic-mismatch-cost",
        type=float,
        default=SEMANTIC_MISMATCH_COST,
        help="Cost for semantic mismatches (default: %(default)s)",
    )
    parser.add_argument(
        "--structural-mismatch-cost",
        type=float,
        default=STRUCTURAL_MISMATCH_COST,
        help="Cost for structural mismatches (default: %(default)s)",
    )
    parser.add_argument(
        "--span-distance-weight",
        type=float,
        default=SPAN_DISTANCE_WEIGHT,
        help="Weight for span distance in similarity calculation (default: %(default)s)",
    )

    args = parser.parse_args()
    old_path, new_path = args.files

    SPAN_THRESHOLD = args.span_threshold
    SEMANTIC_MISMATCH_COST = args.semantic_mismatch_cost
    STRUCTURAL_MISMATCH_COST = args.structural_mismatch_cost
    SPAN_DISTANCE_WEIGHT = args.span_distance_weight

    # Verify files exist
    for path in [old_path, new_path]:
        if not os.path.exists(path):
            print(f"Error: '{path}' does not exist", file=sys.stderr)
            sys.exit(2)

    try:
        # Parse both files
        with open(old_path, "r") as f:
            old_content = f.read()
        with open(new_path, "r") as f:
            new_content = f.read()

        print(f"Parsing {old_path}...")
        old_parser = SimpleSExpParser(old_content)
        old_nodes, old_edges, old_line_map = old_parser.parse()
        print(f"Found {len(old_nodes)} nodes, {len(old_edges)} edges")

        print(f"Parsing {new_path}...")
        new_parser = SimpleSExpParser(new_content)
        new_nodes, new_edges, new_line_map = new_parser.parse()
        print(f"Found {len(new_nodes)} nodes, {len(new_edges)} edges")

        # Generate diff using fuzzy matching
        print(f"\n--- {old_path}")
        print(f"+++ {new_path}")

        diff_lines = fuzzy_graph_diff(old_nodes, new_nodes, old_edges, new_edges)

        if diff_lines:
            for line in diff_lines:
                print(line)
            sys.exit(1)  # Differences found
        else:
            print("Files are identical")
            sys.exit(0)  # Files are identical

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
