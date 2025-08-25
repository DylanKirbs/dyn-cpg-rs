use super::{Cpg, ListenerType, NodeId};
use std::collections::HashMap;
use strum_macros::Display;

#[derive(Debug, Clone, Display, PartialEq, Eq, Hash)]
pub enum EdgeType {
    Unknown,

    // AST
    SyntaxChild,
    SyntaxSibling,

    // Control Flow
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
            tests::{create_test_cpg, create_test_node},
            DescendantTraversal, Edge, EdgeQuery, EdgeType, NodeType,
        },
        desc_trav,
    };
    use std::collections::HashMap;

    #[test]
    fn test_complex_edge_query() {
        let mut cpg = create_test_cpg();
        let node1 = cpg.add_node(
            create_test_node(NodeType::Function {
                name_traversal: desc_trav![],
                name: Some("main".to_string()),
            }),
            0,
            1,
        );
        let node2 = cpg.add_node(create_test_node(NodeType::Identifier), 1, 2);
        let node3 = cpg.add_node(create_test_node(NodeType::Identifier), 2, 3);

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
            create_test_node(NodeType::Function {
                name_traversal: desc_trav![],
                name: Some("main".to_string()),
            }),
            0,
            1,
        );
        let node2 = cpg.add_node(create_test_node(NodeType::Identifier), 1, 2);
        let node3 = cpg.add_node(create_test_node(NodeType::Identifier), 2, 3);

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
            create_test_node(NodeType::Function {
                name_traversal: desc_trav![],
                name: Some("main".to_string()),
            }),
            0,
            1,
        );
        let node2 = cpg.add_node(create_test_node(NodeType::Identifier), 1, 2);
        let node3 = cpg.add_node(create_test_node(NodeType::Statement), 2, 3);

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
