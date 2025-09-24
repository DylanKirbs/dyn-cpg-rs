use crate::cpg::spatial_index::SpatialIndex;

use super::{Cpg, EdgeId, ListenerType, NodeId};
use std::collections::{HashMap, HashSet};
use strum_macros::Display;
use tracing::warn;

#[derive(Debug, Clone, Display, PartialEq, Eq, Hash)]
pub enum EdgeType {
    Unknown,

    // AST
    SyntaxChild,
    SyntaxSibling,

    // Control Flow
    ControlFlowFunctionReturn,
    ControlFlowEpsilon,
    ControlFlowTrue,
    ControlFlowFalse,

    // Program Dependence
    PDControlTrue,
    PDControlFalse,
    PDData(String), // Identifier for data dependencies

    // Listener
    Listener(ListenerType),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Edge {
    pub from: NodeId,
    pub to: NodeId,
    pub type_: EdgeType,
    pub properties: HashMap<String, String>, // As little as possible should be stored here
}

// --- CPG --- ///

impl Cpg {
    /// Get the number of edges in the CPG
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn add_edge(&mut self, edge: Edge) -> EdgeId {
        let id = self.edges.insert(edge.clone());
        self.incoming.entry(edge.to).or_default().insert(id);
        self.outgoing.entry(edge.from).or_default().insert(id);
        id
    }

    pub fn remove_edge(&mut self, edge: EdgeId) -> Option<Edge> {
        if let Some(e) = self.edges.remove(edge) {
            if let Some(in_edges) = self.incoming.get_mut(&e.to) {
                in_edges.remove(&edge);
            }
            if let Some(out_edges) = self.outgoing.get_mut(&e.from) {
                out_edges.remove(&edge);
            }
            Some(e)
        } else {
            None
        }
    }

    pub fn get_incoming_edges(&self, to: NodeId) -> Vec<&Edge> {
        self.incoming
            .get(&to)
            .into_iter()
            .flat_map(|ids| ids.iter().map(|id| &self.edges[*id]))
            .collect()
    }

    pub fn get_outgoing_edges(&self, from: NodeId) -> Vec<&Edge> {
        self.outgoing
            .get(&from)
            .into_iter()
            .flat_map(|ids| ids.iter().map(|id| &self.edges[*id]))
            .collect()
    }

    pub fn get_deterministic_sorted_outgoing_edges(&self, from: NodeId) -> Vec<&Edge> {
        let mut edges = self.get_outgoing_edges(from);
        edges.sort_by(|a, b| {
            if a == b {
                return std::cmp::Ordering::Equal;
            }

            // Sort first on from node span (start)
            let a_from_start = self
                .spatial_index
                .get_node_span(a.from)
                .map(|(s, _)| s)
                .unwrap_or(usize::MAX);
            let b_from_start = self
                .spatial_index
                .get_node_span(b.from)
                .map(|(s, _)| s)
                .unwrap_or(usize::MAX);

            let ord = a_from_start.cmp(&b_from_start);
            if ord != std::cmp::Ordering::Equal {
                return ord;
            }

            // Sort first on from node span (end)
            let a_from_end = self
                .spatial_index
                .get_node_span(a.from)
                .map(|(_, e)| e)
                .unwrap_or(usize::MAX);
            let b_from_end = self
                .spatial_index
                .get_node_span(b.from)
                .map(|(_, e)| e)
                .unwrap_or(usize::MAX);

            let ord = a_from_end.cmp(&b_from_end);
            if ord != std::cmp::Ordering::Equal {
                return ord;
            }

            // Sort first on to node span (start)
            let a_to_start = self
                .spatial_index
                .get_node_span(a.to)
                .map(|(s, _)| s)
                .unwrap_or(usize::MAX);
            let b_to_start = self
                .spatial_index
                .get_node_span(b.to)
                .map(|(s, _)| s)
                .unwrap_or(usize::MAX);

            let ord = a_to_start.cmp(&b_to_start);
            if ord != std::cmp::Ordering::Equal {
                return ord;
            }

            // Sort first on to node span (end)
            let a_to_end = self
                .spatial_index
                .get_node_span(a.to)
                .map(|(_, e)| e)
                .unwrap_or(usize::MAX);
            let b_to_end = self
                .spatial_index
                .get_node_span(b.to)
                .map(|(_, e)| e)
                .unwrap_or(usize::MAX);

            let ord = a_to_end.cmp(&b_to_end);
            if ord != std::cmp::Ordering::Equal {
                return ord;
            }

            // Then on edge types
            let a_type = a.type_.label();
            let b_type = b.type_.label();
            let ord = a_type.cmp(&b_type);
            if ord != std::cmp::Ordering::Equal {
                return ord;
            }

            // Then origin node type
            let a_node_type = self
                .get_node_by_id(&a.from)
                .map(|n| n.type_.label())
                .unwrap_or_default();
            let b_node_type = self
                .get_node_by_id(&b.from)
                .map(|n| n.type_.label())
                .unwrap_or_default();
            let ord = a_node_type.cmp(&b_node_type);
            if ord != std::cmp::Ordering::Equal {
                return ord;
            }

            // Then arrival node type
            let a_node_type = self
                .get_node_by_id(&a.to)
                .map(|n| n.type_.label())
                .unwrap_or_default();
            let b_node_type = self
                .get_node_by_id(&b.to)
                .map(|n| n.type_.label())
                .unwrap_or_default();
            let ord = a_node_type.cmp(&b_node_type);
            if ord != std::cmp::Ordering::Equal {
                return ord;
            }


            warn!(
                "[EDGE SORT] Used unstable node id as tie breaker for edges {:?} ({:?}, {:?})->({:?}, {:?}) and {:?} ({:?}, {:?})->({:?}, {:?}).",
                a, a_from_start, a_from_end, a_to_start, a_to_end,
                b, b_from_start, b_from_end, b_to_start, b_to_end
            );
            let a_id = a.to.as_str();
            let b_id = b.to.as_str();
            a_id.cmp(&b_id)
        });
        edges
    }

    /// Get all of the Syntax Children of a node, ordered by their SyntaxSibling edges
    /// (i.e. in the order they appear in the source code)
    pub fn ordered_syntax_children(&self, root: NodeId) -> Vec<NodeId> {
        // Guard against no edges
        let outgoing_edges = self.get_deterministic_sorted_outgoing_edges(root);
        if outgoing_edges.is_empty() {
            return Vec::new();
        }

        let mut child_nodes = Vec::new();
        let mut sibling_map = HashMap::new();
        let mut has_incoming_sibling = HashSet::new();

        for edge in outgoing_edges {
            if edge.type_ == EdgeType::SyntaxChild {
                child_nodes.push(edge.to);
            }
        }

        // Guard against no children
        if child_nodes.is_empty() {
            return Vec::new();
        }

        for &child in &child_nodes {
            let child_outgoing = self.get_deterministic_sorted_outgoing_edges(child);
            for edge in child_outgoing {
                if edge.type_ == EdgeType::SyntaxSibling {
                    sibling_map.insert(edge.from, edge.to);
                    has_incoming_sibling.insert(edge.to);
                    break; // Each node has at most one outgoing sibling edge
                }
            }
        }

        let start = child_nodes
            .iter()
            .find(|&&node| !has_incoming_sibling.contains(&node))
            .copied();

        let mut ordered = Vec::with_capacity(child_nodes.len());
        let mut current = start;
        while let Some(id) = current {
            ordered.push(id);
            current = sibling_map.get(&id).copied();
        }

        ordered
    }
}

impl EdgeType {
    pub fn colour(&self) -> &'static str {
        match self {
            EdgeType::Unknown => "black",
            EdgeType::SyntaxChild => "blue",
            EdgeType::SyntaxSibling => "green",
            EdgeType::ControlFlowEpsilon => "red",
            EdgeType::ControlFlowFunctionReturn => "darkred",
            EdgeType::ControlFlowTrue => "orange",
            EdgeType::ControlFlowFalse => "purple",
            EdgeType::PDControlTrue => "cyan",
            EdgeType::PDControlFalse => "magenta",
            EdgeType::PDData(_) => "brown",
            EdgeType::Listener(_) => "gray",
        }
    }

    pub fn label(&self) -> String {
        format!("{:?}", self)
            .replace("EdgeType::", "")
            .replace('_', " ")
            .replace('"', "'")
    }
}

// --- Edge query --- //

pub struct EdgeQuery<'a> {
    from: Option<&'a NodeId>,
    to: Option<&'a NodeId>,
    type_: Option<&'a EdgeType>,
}

impl<'query> EdgeQuery<'query> {
    pub fn new() -> Self {
        Self {
            from: None,
            to: None,
            type_: None,
        }
    }

    pub fn from<'id>(mut self, from: &'id NodeId) -> Self
    where
        'id: 'query,
    {
        self.from = Some(from);
        self
    }

    pub fn to<'id>(mut self, to: &'id NodeId) -> Self
    where
        'id: 'query,
    {
        self.to = Some(to);
        self
    }

    pub fn edge_type(mut self, ty: &'query EdgeType) -> Self {
        self.type_ = Some(ty);
        self
    }

    pub fn query(self, graph: &'query Cpg) -> Vec<&'query Edge> {
        graph
            .edges
            .values()
            .filter(|edge| {
                self.from.is_none_or(|f| edge.from == *f)
                    && self.to.is_none_or(|t| edge.to == *t)
                    && self.type_.is_none_or(|t| &edge.type_ == t)
            })
            .collect()
    }
}

impl<'a> Default for EdgeQuery<'a> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {

    use crate::{
        cpg::{
            DescendantTraversal, Edge, EdgeQuery, EdgeType, IdenType, NodeType,
            tests::{create_test_cpg, create_test_node},
        },
        desc_trav,
    };
    use std::collections::HashMap;

    #[test]
    fn test_complex_edge_query() {
        let mut cpg = create_test_cpg();
        let node1 = cpg.add_node(
            create_test_node(
                NodeType::Function {
                    name_traversals: vec![desc_trav![]],
                },
                Some("main".to_string()),
            ),
            0,
            1,
        );
        let node2 = cpg.add_node(
            create_test_node(
                NodeType::Identifier {
                    type_: IdenType::UNKNOWN,
                },
                None,
            ),
            1,
            2,
        );
        let node3 = cpg.add_node(
            create_test_node(
                NodeType::Identifier {
                    type_: IdenType::UNKNOWN,
                },
                None,
            ),
            2,
            3,
        );

        let edge1 = Edge {
            from: node1,
            to: node2,
            type_: EdgeType::SyntaxChild,
            properties: HashMap::new(),
        };
        let edge2 = Edge {
            from: node1,
            to: node3,
            type_: EdgeType::ControlFlowTrue,
            properties: HashMap::new(),
        };
        cpg.add_edge(edge1);
        cpg.add_edge(edge2);

        let query = EdgeQuery::new()
            .from(&node1)
            .edge_type(&EdgeType::SyntaxChild);
        let results = query.query(&cpg);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].to, node2);
    }

    #[test]
    fn test_all_incoming_edges() {
        let mut cpg = create_test_cpg();
        let node1 = cpg.add_node(
            create_test_node(
                NodeType::Function {
                    name_traversals: vec![desc_trav![]],
                },
                Some("main".to_string()),
            ),
            0,
            1,
        );
        let node2 = cpg.add_node(
            create_test_node(
                NodeType::Identifier {
                    type_: IdenType::UNKNOWN,
                },
                None,
            ),
            1,
            2,
        );
        let node3 = cpg.add_node(
            create_test_node(
                NodeType::Identifier {
                    type_: IdenType::UNKNOWN,
                },
                None,
            ),
            2,
            3,
        );

        let edge1 = Edge {
            from: node1,
            to: node2,
            type_: EdgeType::SyntaxChild,
            properties: HashMap::new(),
        };
        let edge2 = Edge {
            from: node3,
            to: node2,
            type_: EdgeType::ControlFlowTrue,
            properties: HashMap::new(),
        };
        cpg.add_edge(edge1);
        cpg.add_edge(edge2);

        let query = EdgeQuery::new().to(&node2);
        let results = query.query(&cpg);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].from, node1);
        assert_eq!(results[1].from, node3);
    }

    #[test]
    fn test_edge_query_combinations() {
        let mut cpg = create_test_cpg();
        let node1 = cpg.add_node(
            create_test_node(
                NodeType::Function {
                    name_traversals: vec![desc_trav![]],
                },
                Some("main".to_string()),
            ),
            0,
            1,
        );
        let node2 = cpg.add_node(
            create_test_node(
                NodeType::Identifier {
                    type_: IdenType::UNKNOWN,
                },
                None,
            ),
            1,
            2,
        );
        let node3 = cpg.add_node(create_test_node(NodeType::Statement, None), 2, 3);

        // Add edges with properties
        let mut edge_props = HashMap::new();
        edge_props.insert("weight".to_string(), "high".to_string());

        let edge1 = Edge {
            from: node1,
            to: node2,
            type_: EdgeType::SyntaxChild,
            properties: edge_props.clone(),
        };
        let edge2 = Edge {
            from: node2,
            to: node3,
            type_: EdgeType::SyntaxChild,
            properties: HashMap::new(),
        };
        cpg.add_edge(edge1);
        cpg.add_edge(edge2);

        // Query with multiple criteria
        let all_syntax_child = EdgeQuery::new()
            .edge_type(&EdgeType::SyntaxChild)
            .query(&cpg);
        assert_eq!(all_syntax_child.len(), 2);

        // Query specific edge
        let specific = EdgeQuery::new().from(&node1).to(&node2).query(&cpg);
        assert_eq!(specific.len(), 1);
        assert_eq!(specific[0].properties, edge_props);
    }
}
