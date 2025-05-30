//! # The CPG (Code Property Graph) module
//! This module provides functionality to generate and update a Code Property Graph (CPG) from source code files.
//! As well as serialize and deserialize the CPG to and from a Gremlin database.

use gremlin_client::GID::String as GIDString;
use gremlin_client::GremlinClient;
use gremlin_client::structure::GValue;
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

/// Trait for serializing and deserializing objects to and from a Gremlin-compatible format.
pub trait GremlinSerializable {
    /// Returns the Gremlin label and the properties as key-value pairs.
    fn serialize(&self) -> Result<(String, Vec<(&str, String)>), CpgError>;

    /// Reconstructs a struct from a Gremlin GValue (typically a vertex or edge).
    fn deserialize(value: &GValue) -> Result<Self, CpgError>
    where
        Self: Sized;
}

// The graph structure for the CPG

#[derive(Debug, Clone, Display)]
pub enum NodeType {
    Unknown,
    Error,
    TranslationUnit,
    Method,
}

#[derive(Debug, Clone)]
pub struct Node {
    pub id: String,
    pub type_: NodeType,
    pub properties: HashMap<String, String>,
}

#[derive(Debug, Clone)]
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
}

#[derive(Debug, Clone)]
pub struct Edge {
    pub from: String,
    pub to: String,
    pub type_: EdgeType,
    pub properties: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub enum ListenerType {
    SyntaxExample,
    ControlFlowExample,
    ProgramDependenceExample,
    // Custom(String), // For custom listeners (may be useful?)
}

#[derive(Debug, Clone)]
pub struct Listener {
    pub id: String,
    pub type_: ListenerType, // Type of listener, disambiguates whether this listens to nodes or edges
    pub source: String,      // The edge or node this listener is associated with
    pub target: String,      // The node this listener is listening to
                             // pub properties: HashMap<String, String>, // if needed
}

#[derive(Debug, Clone)]
pub struct Cpg {
    pub nodes: HashMap<String, Node>,
    pub edges: HashMap<String, Edge>,
    pub listeners: HashMap<String, Listener>,
}

// Functionality to interact with the CPG

impl From<&str> for NodeType {
    fn from(label: &str) -> Self {
        match label {
            "Unknown" => NodeType::Unknown,
            "Error" => NodeType::Error,
            "TranslationUnit" => NodeType::TranslationUnit,
            "Method" => NodeType::Method,
            _ => NodeType::Unknown, // Default case
        }
    }
}

impl From<&str> for EdgeType {
    fn from(label: &str) -> Self {
        match label {
            "SyntaxChild" => EdgeType::SyntaxChild,
            "SyntaxSibling" => EdgeType::SyntaxSibling,
            "ControlFlowEpsilon" => EdgeType::ControlFlowEpsilon,
            "ControlFlowTrue" => EdgeType::ControlFlowTrue,
            "ControlFlowFalse" => EdgeType::ControlFlowFalse,
            "PDControlTrue" => EdgeType::PDControlTrue,
            "PDControlFalse" => EdgeType::PDControlFalse,
            _ if label.starts_with("PDData") => EdgeType::PDData(label.to_string()),
            _ => EdgeType::Unknown, // Default case
        }
    }
}

impl From<&str> for ListenerType {
    fn from(label: &str) -> Self {
        match label {
            "SyntaxExample" => ListenerType::SyntaxExample,
            "ControlFlowExample" => ListenerType::ControlFlowExample,
            "ProgramDependenceExample" => ListenerType::ProgramDependenceExample,
            _ => ListenerType::SyntaxExample, // Default case
        }
    }
}

impl GremlinSerializable for Node {
    fn serialize(&self) -> Result<(String, Vec<(&str, String)>), CpgError> {
        let mut props = vec![
            ("iden", self.id.clone()),
            ("type", format!("{:?}", self.type_)),
        ];

        for (k, v) in &self.properties {
            props.push((k.as_str(), v.clone()));
        }

        Ok((
            "g.addV(iden).property(\"type\", type).next()".to_string(),
            props,
        ))
    }

    fn deserialize(value: &GValue) -> Result<Self, CpgError> {
        // Guard clause to ensure we have a vertex
        if let GValue::Vertex(vertex) = value {
            let id = match vertex.id() {
                GIDString(a) => a.to_string(),
                _ => return Err(CpgError::MissingField("id".to_string())),
            };
            let type_ = NodeType::from(vertex.label().as_str());

            let mut properties = HashMap::new();
            for prop in vertex.iter() {
                let (key, val) = prop;
                let v = val
                    .iter()
                    .next()
                    .ok_or_else(|| CpgError::MissingField(key.clone()))?
                    .value()
                    .clone();
                match v {
                    GValue::String(s) => properties.insert(key.clone(), s),
                    GValue::Int32(i) => properties.insert(key.clone(), i.to_string()),
                    GValue::Float(f) => properties.insert(key.clone(), f.to_string()),
                    GValue::Bool(b) => properties.insert(key.clone(), b.to_string()),
                    _ => {
                        return Err(CpgError::ConversionError(
                            "Unsupported property type".to_string(),
                        ));
                    }
                };
            }

            Ok(Node {
                id,
                type_,
                properties,
            })
        } else {
            Err(CpgError::InvalidFormat("Expected a vertex".to_string()))
        }
    }
}

impl GremlinSerializable for Edge {
    fn serialize(&self) -> Result<(String, Vec<(&str, String)>), CpgError> {
        let mut props = vec![
            ("from", self.from.clone()),
            ("to", self.to.clone()),
            ("type", format!("{:?}", self.type_)),
        ];

        for (k, v) in &self.properties {
            props.push((k.as_str(), v.clone()));
        }

        Ok((
            "g.addE(from).to(g.V(to)).property(\"type\", type).next()".to_string(),
            props,
        ))
    }

    fn deserialize(value: &GValue) -> Result<Self, CpgError> {
        // Guard clause to ensure we have an edge
        if let GValue::Edge(edge) = value {
            let type_ = EdgeType::from(edge.label().as_str());

            let mut properties = HashMap::new();
            for prop in edge.iter() {
                let (key, val) = prop;
                let v = val.value().clone();
                match v {
                    GValue::String(s) => properties.insert(key.clone(), s),
                    GValue::Int32(i) => properties.insert(key.clone(), i.to_string()),
                    GValue::Float(f) => properties.insert(key.clone(), f.to_string()),
                    GValue::Bool(b) => properties.insert(key.clone(), b.to_string()),
                    _ => {
                        return Err(CpgError::ConversionError(
                            "Unsupported property type".to_string(),
                        ));
                    }
                };
            }

            let from = "0".to_string(); // TODO
            let to = "0".to_string(); // TODO

            Ok(Edge {
                from,
                to,
                type_,
                properties,
            })
        } else {
            Err(CpgError::InvalidFormat("Expected an edge".to_string()))
        }
    }
}

impl GremlinSerializable for Listener {
    // Just pretend it's an edge for now
    fn serialize(&self) -> Result<(String, Vec<(&str, String)>), CpgError> {
        let props = vec![
            ("iden", self.id.clone()),
            ("type", format!("{:?}", self.type_)),
            ("source", self.source.clone()),
            ("target", self.target.clone()),
        ];

        Ok((
            "g.addE(iden).to(g.V(target)).property(\"type\", type).property(\"source\", source).next()".to_string(),
            props,
        ))
    }

    fn deserialize(value: &GValue) -> Result<Self, CpgError>
    where
        Self: Sized,
    {
        Err(CpgError::InvalidFormat(
            "Listeners are not yet supported".to_string(),
        ))
    }
}

impl Cpg {
    pub fn new() -> Self {
        Cpg {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            listeners: HashMap::new(),
        }
    }

    pub fn read_and_deserialize(client: &GremlinClient) -> Result<Self, CpgError> {
        let mut cpg = Cpg::new();

        // TODO

        Ok(cpg)
    }

    pub fn add_node(&mut self, node: Node) {
        self.nodes.insert(node.id.clone(), node);
    }

    pub fn add_edge(&mut self, edge: Edge) {
        self.edges
            .insert(format!("{}-{}", edge.from, edge.to), edge);
    }

    pub fn add_listener(&mut self, listener: Listener) {
        self.listeners.insert(listener.id.clone(), listener);
    }

    pub fn get_node(&self, id: &str) -> Option<&Node> {
        self.nodes.get(id)
    }

    pub fn get_edge(&self, from: &str, to: &str) -> Option<&Edge> {
        self.edges.get(&format!("{}-{}", from, to))
    }

    pub fn get_listener(&self, id: &str) -> Option<&Listener> {
        self.listeners.get(id)
    }

    pub fn serialize_and_write(&self, client: &GremlinClient) -> Result<(), CpgError> {
        for node in self.nodes.values() {
            let (label, props) = node.serialize()?;
            client
                .execute(
                    label,
                    &props
                        .iter()
                        .map(|(k, v)| (*k, v as &dyn gremlin_client::ToGValue))
                        .collect::<Vec<_>>(),
                )
                .map_err(|e| CpgError::QueryExecutionError(e.to_string()))?;
        }

        for edge in self.edges.values() {
            let (label, props) = edge.serialize()?;
            client
                .execute(
                    label,
                    &props
                        .iter()
                        .map(|(k, v)| (*k, v as &dyn gremlin_client::ToGValue))
                        .collect::<Vec<_>>(),
                )
                .map_err(|e| CpgError::QueryExecutionError(e.to_string()))?;
        }

        for listener in self.listeners.values() {
            let (label, props) = listener.serialize()?;
            client
                .execute(
                    label,
                    &props
                        .iter()
                        .map(|(k, v)| (*k, v as &dyn gremlin_client::ToGValue))
                        .collect::<Vec<_>>(),
                )
                .map_err(|e| CpgError::QueryExecutionError(e.to_string()))?;
        }

        Ok(())
    }
}

// Unit tests for the CPG module

#[cfg(test)]
mod tests {
    use gremlin_client::ConnectionOptions;

    use super::*;

    fn mk_test_cpg() -> Cpg {
        let mut node_properties = HashMap::new();
        node_properties.insert("name".to_string(), "main".to_string());
        let node = Node {
            id: "node1".to_string(),
            type_: NodeType::Method,
            properties: node_properties,
        };

        let mut edge_properties = HashMap::new();
        edge_properties.insert("condition".to_string(), "true".to_string());
        let edge = Edge {
            from: "node1".to_string(),
            to: "node2".to_string(),
            type_: EdgeType::ControlFlowTrue,
            properties: edge_properties,
        };

        let mut listener_properties = HashMap::new();
        listener_properties.insert("event".to_string(), "onEnter".to_string());
        let listener = Listener {
            id: "listener1".to_string(),
            type_: ListenerType::ControlFlowExample,
            source: "node1".to_string(),
            target: "node2".to_string(),
        };

        let mut cpg = Cpg {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            listeners: HashMap::new(),
        };

        cpg.nodes.insert(node.id.clone(), node);
        cpg.edges.insert(format!("{}-{}", edge.from, edge.to), edge);
        cpg.listeners.insert(listener.id.clone(), listener);

        cpg
    }

    fn connect_to_client() -> gremlin_client::GremlinClient {
        // Mock connection to a Gremlin client
        gremlin_client::GremlinClient::connect(
            ConnectionOptions::builder()
                .host("localhost")
                .port(8182)
                .pool_connection_timeout(Some(std::time::Duration::from_secs(5)))
                .build(),
        )
        .expect("Failed to connect to Gremlin server")
    }

    #[test]
    fn test_gremlin_serialization() {
        // Set up a mock gremlin client and test that we deserialize correctly

        let client = connect_to_client();

        let cpg = mk_test_cpg();
        cpg.serialize_and_write(&client)
            .expect("Failed to serialize and write CPG");

        let new_cpg =
            Cpg::read_and_deserialize(&client).expect("Failed to read and deserialize CPG");

        assert_eq!(new_cpg.nodes.len(), cpg.nodes.len());
    }
}
