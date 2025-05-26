//! # The CPG (Code Property Graph) module
//! This module provides functionality to generate and update a Code Property Graph (CPG) from source code files.
//! As well as serialize and deserialize the CPG to and from a Gremlin database.
use gremlin_client::GValue;
use gremlin_client::process::traversal::ByteCode;
use std::collections::HashMap;

// For now, this is a mock implementation, highly subject to change
#[derive(Debug, Clone)]
pub struct CpgNode {
    pub id: String,
    pub label: String,
    pub properties: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct CpgEdge {
    pub from: String,
    pub to: String,
    pub label: String,
}

#[derive(Debug, Default)]
pub struct MockCpg {
    pub nodes: Vec<CpgNode>,
    pub edges: Vec<CpgEdge>,
}

impl MockCpg {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_node(&mut self, id: &str, label: &str, props: &[(&str, &str)]) {
        let mut properties = HashMap::new();
        for (k, v) in props {
            properties.insert(k.to_string(), v.to_string());
        }
        self.nodes.push(CpgNode {
            id: id.to_string(),
            label: label.to_string(),
            properties,
        });
    }

    pub fn add_edge(&mut self, from: &str, to: &str, label: &str) {
        self.edges.push(CpgEdge {
            from: from.to_string(),
            to: to.to_string(),
            label: label.to_string(),
        });
    }
}

#[derive(Debug)]
pub enum CpgError {
    InvalidFormat(String),
    MissingField(String),
    ConversionError(String),
}

impl std::fmt::Display for CpgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CpgError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
            CpgError::MissingField(field) => write!(f, "Missing required field: {}", field),
            CpgError::ConversionError(msg) => write!(f, "Conversion error: {}", msg),
        }
    }
}

impl std::error::Error for CpgError {}

impl TryFrom<GValue> for CpgNode {
    type Error = CpgError;

    fn try_from(value: GValue) -> Result<Self, Self::Error> {
        match value {
            GValue::Vertex(vertex) => {
                let id = format!("{:?}", vertex.id());
                let label = vertex.label().to_string();

                let properties = HashMap::new();
                // TODO: Currently ignored

                Ok(CpgNode {
                    id,
                    label,
                    properties,
                })
            }
            _ => Err(CpgError::InvalidFormat("Expected Vertex".to_string())),
        }
    }
}

impl TryFrom<GValue> for CpgEdge {
    type Error = CpgError;

    fn try_from(value: GValue) -> Result<Self, Self::Error> {
        match value {
            GValue::Edge(edge) => {
                let from = format!("{:?}", edge.out_v());
                let to = format!("{:?}", edge.in_v());
                let label = edge.label().to_string();

                Ok(CpgEdge { from, to, label })
            }
            _ => Err(CpgError::InvalidFormat("Expected Edge".to_string())),
        }
    }
}

impl TryFrom<GValue> for MockCpg {
    type Error = CpgError;

    fn try_from(value: GValue) -> Result<Self, Self::Error> {
        match value {
            GValue::Map(map) => {
                let mut cpg = MockCpg::new();

                // Extract nodes
                if let Some(nodes_value) = map.get("nodes") {
                    match nodes_value {
                        GValue::List(nodes_list) => {
                            for node_value in nodes_list.clone().into_iter() {
                                let node = CpgNode::try_from(node_value.clone())?;
                                cpg.nodes.push(node);
                            }
                        }
                        _ => {
                            return Err(CpgError::InvalidFormat(
                                "nodes should be a list".to_string(),
                            ));
                        }
                    }
                }

                // Extract edges
                if let Some(edges_value) = map.get("edges") {
                    match edges_value {
                        GValue::List(edges_list) => {
                            for edge_value in edges_list.clone().into_iter() {
                                let edge = CpgEdge::try_from(edge_value.clone())?;
                                cpg.edges.push(edge);
                            }
                        }
                        _ => {
                            return Err(CpgError::InvalidFormat(
                                "edges should be a list".to_string(),
                            ));
                        }
                    }
                }

                Ok(cpg)
            }
            GValue::List(list) => {
                // Handle case where the result is a list of mixed vertices and edges
                let mut cpg = MockCpg::new();

                for item in list {
                    match item {
                        GValue::Vertex(_) => {
                            let node = CpgNode::try_from(item)?;
                            cpg.nodes.push(node);
                        }
                        GValue::Edge(_) => {
                            let edge = CpgEdge::try_from(item)?;
                            cpg.edges.push(edge);
                        }
                        _ => {
                            // Skip unknown types or handle them as needed
                            continue;
                        }
                    }
                }

                Ok(cpg)
            }
            _ => Err(CpgError::InvalidFormat("Expected Map or List".to_string())),
        }
    }
}

// Helper methods for serialization back to Gremlin
impl MockCpg {
    /// Convert the CPG to a format suitable for Gremlin queries
    pub fn to_gremlin_insertions(&self) -> Vec<String> {
        let mut queries = Vec::new();

        // Add vertex insertions
        for node in &self.nodes {
            let mut query = format!("g.addV('{}').property(id, '{}')", node.label, node.id);
            for (key, value) in &node.properties {
                query.push_str(&format!(".property('{}', '{}')", key, value));
            }
            queries.push(query);
        }

        // Add edge insertions
        for edge in &self.edges {
            let query = format!(
                "g.V('{}').addE('{}').to(g.V('{}'))",
                edge.from, edge.label, edge.to
            );
            queries.push(query);
        }

        queries
    }

    /// Get a node by ID
    pub fn get_node(&self, id: &str) -> Option<&CpgNode> {
        self.nodes.iter().find(|node| node.id == id)
    }

    /// Get all edges from a node
    pub fn get_outgoing_edges(&self, node_id: &str) -> Vec<&CpgEdge> {
        self.edges
            .iter()
            .filter(|edge| edge.from == node_id)
            .collect()
    }

    /// Get all edges to a node
    pub fn get_incoming_edges(&self, node_id: &str) -> Vec<&CpgEdge> {
        self.edges
            .iter()
            .filter(|edge| edge.to == node_id)
            .collect()
    }
}

impl From<MockCpg> for ByteCode {
    fn from(cpg: MockCpg) -> Self {
        let mut bytecode = ByteCode::new();

        for step in cpg.to_gremlin_insertions() {
            bytecode.add_step(step);
        }

        bytecode
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpg_creation() {
        let mut cpg = MockCpg::new();
        cpg.add_node("1", "Method", &[("name", "main"), ("code", "main()")]);
        cpg.add_node("2", "Parameter", &[("name", "args"), ("type", "String[]")]);
        cpg.add_edge("1", "2", "AST");

        assert_eq!(cpg.nodes.len(), 2);
        assert_eq!(cpg.edges.len(), 1);

        let method_node = cpg.get_node("1").unwrap();
        assert_eq!(method_node.label, "Method");
        assert_eq!(
            method_node.properties.get("name"),
            Some(&"main".to_string())
        );
    }
}
