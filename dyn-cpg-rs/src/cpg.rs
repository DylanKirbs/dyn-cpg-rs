/// # The CPG (Code Property Graph) module
/// This module provides functionality to generate and update a Code Property Graph (CPG) from source code files.
/// As well as serialize and deserialize the CPG to and from a Gremlin database.
use slotmap::{SlotMap, new_key_type};
use std::collections::{BTreeMap, HashMap};
use strum_macros::Display;
use thiserror::Error;
use tracing::debug;
use tree_sitter::Range;

use crate::diff::SourceEdit;

new_key_type! {
    pub struct NodeId;
    pub struct EdgeId;
}

// Spacial indexing

#[derive(Debug, Clone, Default)]
pub struct SpatialIndex {
    map: BTreeMap<(usize, usize), NodeId>,
}

impl SpatialIndex {
    pub fn new() -> Self {
        Self {
            map: BTreeMap::new(),
        }
    }

    pub fn insert(&mut self, start: usize, end: usize, node_id: NodeId) {
        self.map.insert((start, end), node_id);
    }

    pub fn lookup_overlapping(&self, start: usize, end: usize) -> Vec<&NodeId> {
        self.map
            .range(..=(end, usize::MAX))
            .filter(|(key, _)| {
                let (s, e) = *key;
                s < &end && e > &start
            })
            .map(|(_, id)| id)
            .collect()
    }

    pub fn remove_by_node(&mut self, node_id: &NodeId) {
        self.map.retain(|_, v| v != node_id);
    }
}

// Error handling for CPG operations

#[derive(Debug, Display, Error)]
pub enum CpgError {
    InvalidFormat(String),
    MissingField(String),
    ConversionError(String),
    QueryExecutionError(String),
}

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

#[derive(Debug, Clone, PartialEq)]
pub struct Node {
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

#[derive(Debug, Clone, PartialEq)]
pub struct Edge {
    pub from: NodeId,
    pub to: NodeId,
    pub type_: EdgeType,
    pub properties: HashMap<String, String>, // As little as possible should be stored here
}

#[derive(Debug, Clone)]
pub struct Cpg {
    pub nodes: SlotMap<NodeId, Node>,
    pub edges: SlotMap<EdgeId, Edge>,
    pub outgoing: HashMap<NodeId, Vec<EdgeId>>,
    pub spatial_index: SpatialIndex,
}

// Functionality to interact with the CPG

impl Cpg {
    pub fn new() -> Self {
        Cpg {
            nodes: SlotMap::with_key(),
            edges: SlotMap::with_key(),
            outgoing: HashMap::new(),
            spatial_index: SpatialIndex::new(),
        }
    }

    pub fn add_node(&mut self, node: Node, start_byte: usize, end_byte: usize) -> NodeId {
        let node_id = self.nodes.insert(node);
        self.spatial_index.insert(start_byte, end_byte, node_id);

        node_id
    }

    pub fn add_edge(&mut self, edge: Edge) -> EdgeId {
        let id = self.edges.insert(edge.clone());
        self.outgoing.entry(edge.from).or_default().push(id);
        id
    }

    pub fn get_node_by_id(&self, id: &NodeId) -> Option<&Node> {
        self.nodes.get(*id)
    }

    pub fn get_node_by_offsets(&self, start_byte: usize, end_byte: usize) -> Option<&Node> {
        let overlapping_ids = self.spatial_index.lookup_overlapping(start_byte, end_byte);
        overlapping_ids.iter().find_map(|id| self.nodes.get(**id))
    }

    pub fn get_outgoing_edges(&self, from: NodeId) -> Vec<&Edge> {
        self.outgoing
            .get(&from)
            .into_iter()
            .flat_map(|ids| ids.iter().map(|id| &self.edges[*id]))
            .collect()
    }

    /// Incrementally update the CPG from the CST edits
    pub fn incremental_update(
        &mut self,
        edits: Vec<SourceEdit>,
        changed_ranges: impl ExactSizeIterator<Item = Range>,
    ) {
        debug!(
            "Incremental update with {} edits and {} changed ranges",
            edits.len(),
            changed_ranges.len()
        );

        // TODO:
        // 1. Mark everything within the edits and changed ranges as dirty
        // 2. Rehydrate the AST of the CPG based on the dirty nodes
        // 3. Update the Control Flow based on the AST changes
        // 4. Update the Program Dependence based on the Control Flow changes

        for edit in edits {
            debug!("Applying edit: {:?}", edit);
        }
        for range in changed_ranges {
            debug!("Changed range: {:?}", range);
        }
    }

    /// Compare two CPGs for semantic equality
    /// Returns an iterator over the roots of the subtrees that are mismatched (new, different, or missing)
    pub fn compare(&self, other: &Cpg) -> Vec<NodeId> {
        // We only really care that they are semantically equivalent, so if (for example) node ids are different but the nodes are the same, we should still consider them equal.
        // So a Merkle tree is a good way to do this.

        let mut mismatches = Vec::new();

        // Compare nodes

        return mismatches;
    }

    pub fn compute_merkle_hash(&self, node_id: &NodeId) -> Result<blake3::Hash, CpgError> {
        let node = self.get_node_by_id(&node_id).ok_or_else(|| {
            CpgError::InvalidFormat(format!("Node with ID {:?} not found", node_id))
        })?;

        let mut hasher = blake3::Hasher::new();
        hasher.update(node.type_.to_string().as_bytes());

        // Sort property keys to ensure deterministic hash
        let mut props: Vec<_> = node.properties.iter().collect();
        props.sort();
        for (k, v) in props {
            hasher.update(k.as_bytes());
            hasher.update(v.as_bytes());
        }

        // Recurse on children (AST-style edges only, sorted by destination to normalize sibling order)
        let edges = self.get_outgoing_edges(node_id.clone());
        let mut children: Vec<_> = edges
            .iter()
            .filter(|e| matches!(e.type_, EdgeType::SyntaxChild))
            .map(|e| &e.to)
            .collect();
        children.sort();
        for child_id in children {
            let child_hash = self.compute_merkle_hash(child_id)?;
            hasher.update(child_hash.as_bytes());
        }

        Ok(hasher.finalize())
    }
}

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
                self.from.map_or(true, |f| edge.from == f.clone())
                    && self.to.map_or(true, |t| edge.to == t.clone())
                    && self.type_.map_or(true, |t| &edge.type_ == t)
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
    use super::*;
    use crate::diff::incremental_parse;
    use crate::languages::RegisteredLanguage;
    use crate::resource::Resource;

    #[test]
    fn test_cpg_creation() {
        let cpg = Cpg::new();
        assert!(cpg.nodes.is_empty());
        assert!(cpg.edges.is_empty());
    }

    #[test]
    fn test_add_node() {
        let mut cpg = Cpg::new();
        let node = Node {
            type_: NodeType::TranslationUnit,
            properties: HashMap::new(),
        };
        let node_id = cpg.add_node(node.clone(), 0, 10);
        assert_eq!(cpg.get_node_by_id(&node_id), Some(&node));
    }

    #[test]
    fn test_add_edge() {
        let mut cpg = Cpg::new();
        let node_id1 = cpg.add_node(
            Node {
                type_: NodeType::Function,
                properties: HashMap::new(),
            },
            0,
            1,
        );
        let node_id2 = cpg.add_node(
            Node {
                type_: NodeType::Identifier,
                properties: HashMap::new(),
            },
            1,
            2,
        );
        let edge = Edge {
            from: node_id1,
            to: node_id2,
            type_: EdgeType::SyntaxChild,
            properties: HashMap::new(),
        };

        let edge_id = cpg.add_edge(edge.clone());
        assert_eq!(cpg.edges.get(edge_id), Some(&edge));
    }

    #[test]
    fn test_complex_edge_query() {
        let mut cpg = Cpg::new();
        let node1 = cpg.add_node(
            Node {
                type_: NodeType::Function,
                properties: HashMap::new(),
            },
            0,
            1,
        );
        let node2 = cpg.add_node(
            Node {
                type_: NodeType::Identifier,
                properties: HashMap::new(),
            },
            1,
            2,
        );
        let node3 = cpg.add_node(
            Node {
                type_: NodeType::Identifier,
                properties: HashMap::new(),
            },
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
        let mut cpg = Cpg::new();
        let node1 = cpg.add_node(
            Node {
                type_: NodeType::Function,
                properties: HashMap::new(),
            },
            0,
            1,
        );
        let node2 = cpg.add_node(
            Node {
                type_: NodeType::Identifier,
                properties: HashMap::new(),
            },
            1,
            2,
        );
        let node3 = cpg.add_node(
            Node {
                type_: NodeType::Identifier,
                properties: HashMap::new(),
            },
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
    fn test_spatial_index() {
        let mut cpg = Cpg::new();
        let node_id1 = cpg.add_node(
            Node {
                type_: NodeType::Function,
                properties: HashMap::new(),
            },
            0,
            10,
        );
        let node_id2 = cpg.add_node(
            Node {
                type_: NodeType::Identifier,
                properties: HashMap::new(),
            },
            5,
            15,
        );
        let _node_id3 = cpg.add_node(
            Node {
                type_: NodeType::Identifier,
                properties: HashMap::new(),
            },
            12,
            20,
        );

        let overlapping = cpg.spatial_index.lookup_overlapping(8, 12);
        assert_eq!(overlapping.len(), 2);
        assert!(overlapping.contains(&&node_id1));
        assert!(overlapping.contains(&&node_id2));

        let non_overlapping = cpg.spatial_index.lookup_overlapping(21, 25);
        assert!(non_overlapping.is_empty());
        cpg.spatial_index.remove_by_node(&node_id2);
    }

    #[test]
    fn test_incr_reparse() {
        crate::logging::init();

        // Init the lang and parser
        let lang: RegisteredLanguage = "c".parse().expect("Failed to parse language");
        let mut parser = lang.get_parser().expect("Failed to get parser for C");

        // Load sample1.old.c and sample1.new.c from samples/
        let s_orig = Resource::new("samples/sample1.old.c")
            .expect("Failed to create resource for sample1.old.c");
        let s_new = Resource::new("samples/sample1.new.c")
            .expect("Failed to create resource for sample1.new.c");

        // Read the contents of the files
        let old_src = s_orig.read_bytes().expect("Failed to read sample1.old.c");
        let new_src = s_new.read_bytes().expect("Failed to read sample1.new.c");

        // Parse the original file
        let mut old_tree = parser
            .parse(old_src.clone(), None)
            .expect("Failed to parse original file");

        // Parse the new file
        let (edits, new_tree) = incremental_parse(&mut parser, &old_src, &new_src, &mut old_tree)
            .expect("Failed to incrementally parse new file");

        // Changed ranges
        let changed_ranges = old_tree.changed_ranges(&new_tree);
        assert!(changed_ranges.len() != 0, "No changed ranges found");

        let mut cpg = lang
            .cst_to_cpg(old_tree)
            .expect("Failed to convert old tree to CPG");

        cpg.incremental_update(edits, changed_ranges);

        let new_cpg = lang
            .cst_to_cpg(new_tree)
            .expect("Failed to convert new tree to CPG");

        // Check the difference between the two CPGs
        let diff = cpg.compare(&new_cpg);
        assert!(
            diff.is_empty(),
            "CPGs should be semantically equivalent, but found differences: {:?}",
            diff
        );
    }
}
