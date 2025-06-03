/// # The CPG (Code Property Graph) module
/// This module provides functionality to generate and update a Code Property Graph (CPG) from source code files.
/// As well as serialize and deserialize the CPG to and from a Gremlin database.
use std::collections::HashMap;
use strum_macros::Display;

// Error handling for CPG operations

#[derive(Debug)]
pub enum CpgError {
    InvalidFormat(String),
    MissingField(String),
    ConversionError(String),
    QueryExecutionError(String),
}

impl std::fmt::Display for CpgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CpgError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
            CpgError::MissingField(field) => write!(f, "Missing required field: {}", field),
            CpgError::ConversionError(msg) => write!(f, "Conversion error: {}", msg),
            CpgError::QueryExecutionError(msg) => write!(f, "Query execution error: {}", msg),
        }
    }
}

impl std::error::Error for CpgError {}

// The graph structure for the CPG

#[derive(Debug, Clone, Display, PartialEq)]
pub enum NodeType {
    /// Language-implementation specific nodes
    LanguageImplementation(String),

    /// Represents an error in the source code
    Error(String),

    TranslationUnit, // The root of the CPG, representing the entire translation unit (e.g., a file)
    Function,        // A function definition or declaration
    Identifier,      // An identifier (variable, function name, etc.)
    Statement,       // A statement that can be executed
    Expression,      // An expression that can be evaluated
    Type,            // A type definition or usage
    Comment,         // A comment in the source code

    // The weeds (should these be subtypes of statement, expression, etc. or their own types?)
    Call,   // A function call expression
    Return, // A return statement in a function
    Block,  // Compound statement, e.g., a block of code enclosed in braces
}

pub type NodeId = String;

#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub id: NodeId,
    pub type_: NodeType,
    pub properties: HashMap<String, String>, // As little as possible should be stored here
}

#[derive(Debug, Clone, Display, PartialEq)]
pub enum ListenerType {
    Unknown,
    // TODO: Figure out what we need here
}

#[derive(Debug, Clone, Display, PartialEq)]
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

pub type EdgeId = String;

#[derive(Debug, Clone, PartialEq)]
pub struct Edge {
    pub from: NodeId,
    pub to: NodeId,
    pub type_: EdgeType,
    pub properties: HashMap<String, String>, // As little as possible should be stored here
}

#[derive(Debug, Clone)]
pub struct Cpg {
    nodes: HashMap<NodeId, Node>,
    edges: HashMap<NodeId, Vec<Edge>>,
}

// Functionality to interact with the CPG

impl Cpg {
    pub fn new(
        nodes: Option<HashMap<NodeId, Node>>,
        edges: Option<HashMap<NodeId, Vec<Edge>>>,
    ) -> Self {
        Cpg {
            nodes: nodes.unwrap_or_default(),
            edges: edges.unwrap_or_default(),
        }
    }

    pub fn add_node(&mut self, node: Node) {
        self.nodes.insert(node.id.clone(), node);
    }

    pub fn add_edge(&mut self, edge: Edge) {
        self.edges.entry(edge.from.clone()).or_default().push(edge);
    }

    pub fn get_node(&self, id: &NodeId) -> Option<&Node> {
        self.nodes.get(id)
    }

    /// Returns all edges from the given node.
    /// For more complex queries, use `EdgeQuery`.
    pub fn get_outgoing_edges(&self, from: &NodeId) -> Option<&Vec<Edge>> {
        self.edges.get(from)
    }
}

pub struct EdgeQuery<'a> {
    from: Option<&'a NodeId>,
    to: Option<&'a NodeId>,
    type_: Option<&'a EdgeType>,
}

impl<'a> EdgeQuery<'a> {
    pub fn new() -> Self {
        Self {
            from: None,
            to: None,
            type_: None,
        }
    }

    pub fn from(mut self, from: &'a NodeId) -> Self {
        self.from = Some(from);
        self
    }

    pub fn to(mut self, to: &'a NodeId) -> Self {
        self.to = Some(to);
        self
    }

    pub fn edge_type(mut self, ty: &'a EdgeType) -> Self {
        self.type_ = Some(ty);
        self
    }

    pub fn query(self, graph: &Cpg) -> Vec<&Edge> {
        if let Some(from) = self.from {
            graph
                .edges
                .get(from)
                .into_iter()
                .flat_map(move |edges| {
                    edges.iter().filter(move |edge| {
                        self.to.is_none_or(|t| edge.to == t.clone())
                            && self.type_.is_none_or(|ty| edge.type_ == *ty)
                    })
                })
                .collect::<Vec<_>>()
        } else {
            graph
                .edges
                .iter()
                .flat_map(move |(_, edges)| {
                    edges.iter().filter(move |edge| {
                        self.to.is_none_or(|t| edge.to == t.clone())
                            && self.type_.is_none_or(|ty| edge.type_ == *ty)
                    })
                })
                .collect::<Vec<_>>()
        }
    }
}

impl<'a> Default for EdgeQuery<'a> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpg_creation() {
        let cpg = Cpg::new(None, None);
        assert!(cpg.nodes.is_empty());
        assert!(cpg.edges.is_empty());
    }

    #[test]
    fn test_add_node() {
        let mut cpg = Cpg::new(None, None);
        let node = Node {
            id: "node1".to_string(),
            type_: NodeType::TranslationUnit,
            properties: HashMap::new(),
        };
        cpg.add_node(node.clone());
        assert_eq!(cpg.get_node(&"node1".to_string()), Some(&node));
    }

    #[test]
    fn test_add_edge() {
        let mut cpg = Cpg::new(None, None);
        let edge = Edge {
            from: "node1".to_string(),
            to: "node2".to_string(),
            type_: EdgeType::SyntaxChild,
            properties: HashMap::new(),
        };
        cpg.add_edge(edge.clone());
        assert_eq!(
            cpg.get_outgoing_edges(&"node1".to_string()).unwrap().len(),
            1
        );
    }

    #[test]
    fn test_complex_edge_query() {
        let mut cpg = Cpg::new(None, None);
        let edge1 = Edge {
            from: "node1".to_string(),
            to: "node2".to_string(),
            type_: EdgeType::SyntaxChild,
            properties: HashMap::new(),
        };
        let edge2 = Edge {
            from: "node1".to_string(),
            to: "node3".to_string(),
            type_: EdgeType::ControlFlowTrue,
            properties: HashMap::new(),
        };
        cpg.add_edge(edge1);
        cpg.add_edge(edge2);

        let node_id = "node1".to_string();
        let query = EdgeQuery::new()
            .from(&node_id)
            .edge_type(&EdgeType::SyntaxChild);
        let results = query.query(&cpg);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].to, "node2");
    }

    #[test]
    fn test_all_incoming_edges() {
        let mut cpg = Cpg::new(None, None);
        let edge1 = Edge {
            from: "node1".to_string(),
            to: "node2".to_string(),
            type_: EdgeType::SyntaxChild,
            properties: HashMap::new(),
        };
        let edge2 = Edge {
            from: "node3".to_string(),
            to: "node2".to_string(),
            type_: EdgeType::ControlFlowTrue,
            properties: HashMap::new(),
        };
        cpg.add_edge(edge1);
        cpg.add_edge(edge2);

        let node_id = "node2".to_string();
        let query = EdgeQuery::new().to(&node_id);
        let results = query.query(&cpg);
        assert_eq!(results.len(), 2);
    }
}
