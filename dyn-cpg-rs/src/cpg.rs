/// # The CPG (Code Property Graph) module
/// This module provides functionality to generate and update a Code Property Graph (CPG) from source code files.
/// As well as serialize and deserialize the CPG to and from a Gremlin database.
use crate::{
    diff::SourceEdit,
    languages::{RegisteredLanguage, cf_pass, data_dep_pass},
};
use slotmap::{SlotMap, new_key_type};
use std::{
    cmp::{max, min},
    collections::{BTreeMap, HashMap, HashSet},
};
use strum_macros::Display;
use thiserror::Error;
use tracing::debug;
use tree_sitter::Range;

// --- SlotMap Key Types --- //

new_key_type! {
    pub struct NodeId;
    pub struct EdgeId;
}

// --- Helper Functions --- //

fn to_sorted_vec(properties: &HashMap<String, String>) -> Vec<(String, String)> {
    let mut vec: Vec<_> = properties
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    vec.sort_by(|a, b| a.0.cmp(&b.0));
    vec
}

// --- Spacial indexing --- //

#[derive(Debug, Clone, Default)]
pub struct SpatialIndex {
    map: BTreeMap<(usize, usize), NodeId>,
    reverse: HashMap<NodeId, (usize, usize)>,
}

impl SpatialIndex {
    pub fn new() -> Self {
        Self {
            map: BTreeMap::new(),
            reverse: HashMap::new(),
        }
    }

    pub fn insert(&mut self, start: usize, end: usize, node_id: NodeId) {
        self.map.insert((start, end), node_id);
        self.reverse.insert(node_id, (start, end));
    }

    pub fn lookup_nodes_from_range(&self, start: usize, end: usize) -> Vec<&NodeId> {
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
        if let Some(range) = self.reverse.remove(node_id) {
            self.map.remove(&range);
        }
    }

    pub fn get_range_from_node(&self, node_id: &NodeId) -> Option<(usize, usize)> {
        self.reverse.get(node_id).copied()
    }
}

// --- Error handling for CPG operations --- //

#[derive(Debug, Display, Error)]
pub enum CpgError {
    InvalidFormat(String),
    MissingField(String),
    ConversionError(String),
    QueryExecutionError(String),
}

// --- Traversal Aid for Language Agnosticism --- //

#[derive(Debug, Clone, Display, PartialEq, Eq, Hash)]
/// Represents either a named or an offset to a child in the source code.
pub enum ChildReference {
    /// Unknown child, used when the child is not specified or does not exist.
    Unknown,
    /// A field name and kind, used to reference a specific child by its field name and expected kind.
    FieldNameAndKind((String, String)),
    /// Just a field name, used to reference a child by its field name without specifying the kind.
    Field(String),
    /// An offset to a child, used when the exact child is not known but its order with respect to its siblings is known.
    Offset(usize),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Represents a traversal path through the descendants of a node in the syntax tree.
/// Used to navigate from a subtree root, to a specific descendant node.
/// *This aids in the "language agnosticism" of the control flow pass.*
pub struct DescendantTraversal {
    traversal: Vec<ChildReference>,
    cpg_node_id: Option<NodeId>,
}

impl DescendantTraversal {
    pub fn new(traversal: Vec<ChildReference>) -> Self {
        DescendantTraversal {
            traversal,
            cpg_node_id: None,
        }
    }

    pub fn get_ts_descendant(self, node: tree_sitter::Node) -> Option<tree_sitter::Node> {
        let mut current_node = node;
        for step in &self.traversal {
            match step {
                ChildReference::Unknown => return None,
                ChildReference::FieldNameAndKind((field, kind)) => {
                    let mut cursor = current_node.walk();
                    let mut found = false;
                    for child in current_node.children_by_field_name(&field, &mut cursor) {
                        if child.kind() == kind {
                            current_node = child;
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        return None;
                    }
                }
                ChildReference::Field(field) => {
                    if let Some(child) = current_node.child_by_field_name(&field) {
                        current_node = child;
                    } else {
                        return None;
                    }
                }
                ChildReference::Offset(offset) => {
                    if let Some(child) = current_node.child(*offset) {
                        current_node = child;
                    } else {
                        return None;
                    }
                }
            }
        }
        Some(current_node)
    }

    pub fn get_cpg_descendant(
        &self,
        ts_node: tree_sitter::Node,
        cpg_node_id: NodeId,
        cpg: &mut Cpg,
    ) -> Option<NodeId> {
        // Parallel traversal of ts and cpg to get the corresponding cpg node

        let mut current_node = ts_node;
        let mut curr_cpg_node = cpg_node_id;

        for step in &self.traversal {
            match step {
                ChildReference::Unknown => return None,
                ChildReference::FieldNameAndKind((field, kind)) => {
                    let mut cursor = current_node.walk();
                    let mut found = false;
                    for child in current_node.children_by_field_name(&field, &mut cursor) {
                        if child.kind() == kind {
                            current_node = child;
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        return None;
                    }
                    // If found, traverse the cpg node to the ts node
                    let idx = cursor.descendant_index();
                    match cpg.ordered_syntax_children(curr_cpg_node).get(idx) {
                        Some(node_id) => curr_cpg_node = *node_id,
                        None => return None,
                    }
                }
                ChildReference::Field(field) => {
                    let mut cursor = current_node.walk();
                    let mut found = false;
                    for child in current_node.children_by_field_name(&field, &mut cursor) {
                        current_node = child;
                        found = true;
                        break;
                    }
                    if !found {
                        return None;
                    }
                    // If found, traverse the cpg node to the ts node
                    let idx = cursor.descendant_index();
                    match cpg.ordered_syntax_children(curr_cpg_node).get(idx) {
                        Some(node_id) => curr_cpg_node = *node_id,
                        None => return None,
                    }
                }
                ChildReference::Offset(offset) => {
                    if let Some(child) = current_node.child(*offset) {
                        current_node = child;
                    } else {
                        return None;
                    }
                    // If found, traverse the cpg node to the ts node
                    match cpg.ordered_syntax_children(curr_cpg_node).get(*offset) {
                        Some(node_id) => curr_cpg_node = *node_id,
                        None => return None,
                    }
                }
            }
        }

        debug!(
            "Found CPG node {:?} for TS node {:?} with traversal {:?}",
            curr_cpg_node,
            ts_node.id(),
            self.traversal
        );

        Some(curr_cpg_node)
    }

    pub fn get_cpg_node_id(&self) -> Option<NodeId> {
        self.cpg_node_id
    }

    pub fn set_cpg_node_id(&mut self, id: Option<NodeId>) {
        self.cpg_node_id = id;
    }
}

#[macro_export]
/// A macro to create a `DescendantTraversal` from a list of child references.
/// It supports different forms:
/// - Named fields with kind: `("field_name", "kind")`
/// - Offset: `#(3)` [0 indicates the first child]
/// - Just a field name: `"field_name"`
/// Example usage:
/// ```
/// use crate::desc_trav;
/// let traversal = desc_trav![
///   ("declarator", "function_declarator"),
///   #3,
/// ]
/// ```
macro_rules! desc_trav {
    ( $( $x:tt ),* $(,)? ) => {
        DescendantTraversal::new(vec![
            $( desc_trav!(@expand $x) ),*
        ])
    };

    // Tuple form: ("field", "kind")
    (@expand ($field:expr, $kind:expr)) => {
        ChildReference::FieldNameAndKind(($field.to_string(), $kind.to_string()))
    };

    // Offset form: #(3)
    (@expand #($offset:expr)) => {
        ChildReference::Offset($offset)
    };

    // Just a field name: "field"
    (@expand $field:expr) => {
        ChildReference::Field($field.to_string())
    };
}

// --- The graph structure for the CPG --- //

#[derive(Debug, Clone, Display, PartialEq, Eq, Hash)]
pub enum NodeType {
    /// Language-implementation specific nodes
    LanguageImplementation(String),

    /// Represents an error in the source code
    Error(String),

    /// The root of the CPG, representing the entire translation unit (e.g., a file)
    TranslationUnit,

    /// A function definition or declaration
    Function {
        name_traversal: DescendantTraversal,
        /// Optional name, can be derived from the traversal
        name: Option<String>,
    },

    /// An identifier (variable, etc.)
    Identifier,

    /// A statement that can be executed
    Statement,

    /// An expression that can be evaluated
    Expression,

    /// A type definition or usage
    Type,

    /// A comment in the source code
    Comment,

    /// Conditional branching constructs
    Branch {
        condition: DescendantTraversal,
        then_branch: DescendantTraversal,
        else_branch: DescendantTraversal,
    },

    /// Loop constructs
    Loop {
        condition: DescendantTraversal,
        body: DescendantTraversal,
    },

    /// Compound statement, e.g., a block of code enclosed in braces
    Block,

    /// A function call expression
    Call,

    /// A return statement in a function
    Return,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    pub type_: NodeType,
    pub properties: HashMap<String, String>, // As little as possible should be stored here
}

impl Node {
    pub fn update_type(&mut self, type_: NodeType) -> Option<()> {
        if self.type_ != type_ {
            self.type_ = type_;
            Some(())
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Display, PartialEq, Eq, Hash)]
pub enum ListenerType {
    Unknown,
    // TODO: Figure out what we need here
}

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

#[derive(Debug, Clone)]
/// The Code Property Graph (CPG) structure
pub struct Cpg {
    /// The root node of the CPG, if set
    root: Option<NodeId>,
    /// Maps NodeId to Node
    nodes: SlotMap<NodeId, Node>,
    /// Maps EdgeId to Edge
    edges: SlotMap<EdgeId, Edge>,
    /// Maps NodeId to a list of EdgeIds that point to it
    incoming: HashMap<NodeId, Vec<EdgeId>>,
    /// Maps NodeId to a list of EdgeIds that point from it
    outgoing: HashMap<NodeId, Vec<EdgeId>>,
    /// Spatial index for fast lookups by byte range
    spatial_index: SpatialIndex,
    /// The language of the CPG
    language: RegisteredLanguage,
    /// The source that the tree/CPG was parsed from
    source: Vec<u8>,
}

// --- Comparison Results --- //

/// Detailed result of a CPG comparison
#[derive(Debug, Clone, PartialEq)]
pub enum DetailedComparisonResult {
    /// The CPGs are semantically equivalent
    Equivalent,
    /// The CPGs have structural differences
    StructuralMismatch {
        /// Functions present only in the left CPG
        only_in_left: Vec<String>,
        /// Functions present only in the right CPG
        only_in_right: Vec<String>,
        /// Functions that exist in both but have differences
        function_mismatches: Vec<FunctionComparisonResult>,
    },
}

/// Result of comparing a single function between two CPGs
#[derive(Debug, Clone, PartialEq)]
pub enum FunctionComparisonResult {
    /// The functions are equivalent
    Equivalent,
    /// The functions differ
    Mismatch {
        /// The name of the function
        function_name: String,
        /// Details about the mismatch
        details: String,
    },
}

// Functionality to interact with the CPG

impl Cpg {
    pub fn new(lang: RegisteredLanguage, source: Vec<u8>) -> Self {
        Cpg {
            root: None,
            nodes: SlotMap::with_key(),
            edges: SlotMap::with_key(),
            incoming: HashMap::new(),
            outgoing: HashMap::new(),
            spatial_index: SpatialIndex::new(),
            language: lang,
            source: source,
        }
    }

    pub fn set_root(&mut self, root: NodeId) {
        self.root = Some(root);
    }

    pub fn get_root(&self) -> Option<NodeId> {
        self.root
    }

    pub fn get_language(&self) -> &RegisteredLanguage {
        &self.language
    }

    pub fn get_source(&self) -> &Vec<u8> {
        &self.source
    }

    /// Get the number of nodes in the CPG
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of edges in the CPG
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Add a node to the CPG and update the spatial index
    /// If no root is set, the first node added will be assumed to be the root, this can be overridden using `set_root`.
    pub fn add_node(&mut self, node: Node, start_byte: usize, end_byte: usize) -> NodeId {
        let node_id = self.nodes.insert(node);
        self.spatial_index.insert(start_byte, end_byte, node_id);

        if self.root.is_none() {
            self.set_root(node_id);
        }

        node_id
    }

    pub fn add_edge(&mut self, edge: Edge) -> EdgeId {
        let id = self.edges.insert(edge.clone());
        self.incoming.entry(edge.to).or_default().push(id);
        self.outgoing.entry(edge.from).or_default().push(id);
        id
    }

    pub fn get_node_by_id(&self, id: &NodeId) -> Option<&Node> {
        self.nodes.get(*id)
    }

    pub fn get_node_by_id_mut(&mut self, id: &NodeId) -> Option<&mut Node> {
        self.nodes.get_mut(*id)
    }

    pub fn get_node_by_offsets(&self, start_byte: usize, end_byte: usize) -> Vec<&Node> {
        let overlapping_ids = self
            .spatial_index
            .lookup_nodes_from_range(start_byte, end_byte);

        overlapping_ids
            .into_iter()
            .filter_map(|id| self.nodes.get(*id))
            .collect()
    }

    pub fn get_node_ids_by_offsets(&self, start_byte: usize, end_byte: usize) -> Vec<NodeId> {
        self.spatial_index
            .lookup_nodes_from_range(start_byte, end_byte)
            .into_iter()
            .cloned()
            .collect()
    }

    pub fn get_smallest_node_id_containing_range(
        &self,
        start_byte: usize,
        end_byte: usize,
    ) -> Option<NodeId> {
        let overlapping_ids = self
            .spatial_index
            .lookup_nodes_from_range(start_byte, end_byte);
        overlapping_ids
            .into_iter()
            .filter(|id| {
                // Only consider nodes that fully contain the range
                let range = self
                    .spatial_index
                    .get_range_from_node(id)
                    .expect("NodeId should have a range");
                range.0 <= start_byte && range.1 >= end_byte
            })
            .min_by_key(|id| {
                let range = self
                    .spatial_index
                    .get_range_from_node(id)
                    .expect("NodeId should have a range");
                range.1 - range.0
            })
            .cloned()
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

    /// Incrementally update the CPG from the CST edits
    pub fn incremental_update(
        &mut self,
        edits: Vec<SourceEdit>,
        changed_ranges: impl ExactSizeIterator<Item = Range>,
        new_tree: &tree_sitter::Tree,
    ) {
        debug!(
            "Incremental update with {} edits and {} changed ranges",
            edits.len(),
            changed_ranges.len()
        );

        let mut dirty_nodes = HashMap::new();
        for range in changed_ranges {
            debug!("TS Changed range: {:?}", range);
            if let Some(node_id) =
                self.get_smallest_node_id_containing_range(range.start_byte, range.end_byte)
            {
                dirty_nodes.insert(
                    node_id.clone(),
                    (
                        range.start_byte,
                        range.end_byte,
                        range.start_byte,
                        range.end_byte,
                    ),
                );
            } else {
                debug!(
                    "No node found for changed range: {:?}",
                    (range.start_byte, range.end_byte)
                );
            }
        }

        for edit in edits {
            debug!("Textual edit: {:?}", edit);
            if let Some(node_id) =
                self.get_smallest_node_id_containing_range(edit.old_start, edit.old_end)
            {
                dirty_nodes
                    .entry(node_id)
                    .and_modify(|existing_dirty| {
                        *existing_dirty = (
                            min(edit.old_start, existing_dirty.0),
                            max(edit.old_end, existing_dirty.1),
                            min(edit.new_start, existing_dirty.2),
                            max(edit.new_end, existing_dirty.3),
                        );
                    })
                    .or_insert((edit.old_start, edit.old_end, edit.new_start, edit.new_end));
            } else {
                debug!(
                    "No node found for edit range: {:?}",
                    (edit.old_start, edit.old_end)
                );
            }
        }

        debug!("Rehydrating dirty nodes: {:?}", dirty_nodes);
        let mut rehydrated_nodes = Vec::new();
        let mut affected_functions = HashSet::new();

        // Collect all functions that contain dirty nodes before rehydration
        for (id, _) in &dirty_nodes {
            if let Some(function_id) = self.find_containing_function(*id) {
                affected_functions.insert(function_id);
            }
        }

        // Rehydrate dirty nodes
        for (id, pos) in dirty_nodes {
            let new_node = self.rehydrate(id, pos, new_tree);
            match new_node {
                Ok(new_id) => {
                    debug!("Rehydrated node {:?} to new id {:?}", id, new_id);
                    rehydrated_nodes.push(new_id);

                    // Also find the function that contains the new node
                    if let Some(function_id) = self.find_containing_function(new_id) {
                        affected_functions.insert(function_id);
                    }
                }
                Err(e) => {
                    debug!("Failed to rehydrate node {:?}: {}", id, e);
                }
            }
        }

        debug!(
            "Computing control flow for {} affected functions",
            affected_functions.len()
        );
        for function_id in affected_functions.clone() {
            cf_pass(self, function_id)
                .map_err(|e| {
                    debug!(
                        "Failed to recompute control flow for function {:?}: {}",
                        function_id, e
                    )
                })
                .ok();
        }

        debug!(
            "Computing program dependence for {} affected functions",
            affected_functions.len()
        );
        for function_id in affected_functions {
            data_dep_pass(self, function_id)
                .map_err(|e| {
                    debug!(
                        "Failed to recompute data dependence for function {:?}: {}",
                        function_id, e
                    )
                })
                .ok();
        }

        debug!("Incremental update complete");
    }

    /// Find the function node that contains the given node
    fn find_containing_function(&self, node_id: NodeId) -> Option<NodeId> {
        let mut current = node_id;
        let mut visited = HashSet::new();

        // Traverse up the syntax tree looking for a function node
        while !visited.contains(&current) {
            visited.insert(current);

            if let Some(node) = self.get_node_by_id(&current) {
                match node.type_ {
                    NodeType::Function { .. } => {
                        return Some(current);
                    }
                    _ => {}
                }
            }

            // Find parent via SyntaxChild edge
            let parent_edges: Vec<_> = self
                .get_incoming_edges(current)
                .into_iter()
                .filter(|e| e.type_ == EdgeType::SyntaxChild)
                .collect();

            if let Some(parent_edge) = parent_edges.first() {
                current = parent_edge.from;
            } else {
                break;
            }
        }

        None
    }

    fn rehydrate(
        &mut self,
        id: NodeId,
        pos: (usize, usize, usize, usize),
        new_tree: &tree_sitter::Tree,
    ) -> Result<NodeId, CpgError> {
        // Check if the node being rehydrated is the current root
        let is_current_root = self.root == Some(id);

        // Translate the new subtree from the new_tree based on the position
        let mut cursor = new_tree
            .root_node()
            .descendant_for_byte_range(pos.2, pos.3)
            .ok_or(CpgError::MissingField(format!(
                "No subtree found for range {:?} in new tree",
                (pos.2, pos.3)
            )))?
            .walk();

        let new_subtree_root = crate::languages::translate(self, &mut cursor).map_err(|e| {
            CpgError::ConversionError(format!("Failed to translate new subtree: {}", e))
        })?;

        // Link new subtree to existing CPG in place of old subtree
        let old_left_sibling_id = self
            .get_incoming_edges(id)
            .into_iter()
            .find(|e| e.type_ == EdgeType::SyntaxSibling);
        if let Some(sibling_edge) = old_left_sibling_id {
            self.add_edge(Edge {
                from: sibling_edge.from,
                to: new_subtree_root,
                type_: EdgeType::SyntaxSibling,
                properties: sibling_edge.properties.clone(), // Copy properties from the old edge (TODO double check if this is needed)
            });
        }

        let old_right_sibling_id = self
            .get_outgoing_edges(id)
            .into_iter()
            .find(|e| e.type_ == EdgeType::SyntaxSibling);
        if let Some(sibling_edge) = old_right_sibling_id {
            self.add_edge(Edge {
                from: new_subtree_root,
                to: sibling_edge.to,
                type_: EdgeType::SyntaxSibling,
                properties: sibling_edge.properties.clone(), // Copy properties from the old edge (TODO double check if this is needed)
            });
        }

        let old_parent_edge = self
            .get_incoming_edges(id)
            .into_iter()
            .find(|e| e.type_ == EdgeType::SyntaxChild);
        if let Some(parent_edge) = old_parent_edge {
            self.add_edge(Edge {
                from: parent_edge.from,
                to: new_subtree_root,
                type_: EdgeType::SyntaxChild,
                properties: parent_edge.properties.clone(), // Copy properties from the old edge (TODO double check if this is needed)
            });
        } else if is_current_root {
            debug!(
                "Rehydrating root node {:?}, setting new root to {:?}",
                id, new_subtree_root
            );
            self.set_root(new_subtree_root);
        } else {
            debug!(
                "No parent edge found for node {:?}, but it's not the root node - leaving root unchanged",
                id
            );
        }

        // Remove the old subtree from the CPG
        self.remove_subtree(id).map_err(|e| {
            CpgError::ConversionError(format!("Failed to remove old subtree: {}", e))
        })?;

        Ok(new_subtree_root)
    }

    /// Recursively removes a subtree from the CPG by its root node ID
    /// This function now properly cleans up edges associated with the removed nodes.
    pub fn remove_subtree(&mut self, root: NodeId) -> Result<(), CpgError> {
        // 1. Recursively remove child subtrees first
        // Collect edges to avoid borrowing issues
        let child_edges: Vec<_> = self
            .get_outgoing_edges(root)
            .into_iter()
            .filter(|e| e.type_ == EdgeType::SyntaxChild)
            .cloned() // Clone edges to avoid holding references
            .collect();

        for edge in child_edges {
            self.remove_subtree(edge.to)?;
        }

        // 2. Now remove the root node itself and its associated edges
        // Remove the node data and spatial index entry
        self.nodes.remove(root);
        self.spatial_index.remove_by_node(&root);

        // 3. Crucially: Remove all edges connected to this node
        // First collect ALL edge IDs that reference this node from the main edges SlotMap
        let mut edges_to_remove = Vec::new();
        for (edge_id, edge) in self.edges.iter() {
            if edge.from == root || edge.to == root {
                edges_to_remove.push(edge_id);
            }
        }

        // Remove each edge and update adjacency lists
        for edge_id in edges_to_remove {
            if let Some(edge) = self.edges.remove(edge_id) {
                // Remove from outgoing list of the 'from' node
                if let Some(outgoing_list) = self.outgoing.get_mut(&edge.from) {
                    outgoing_list.retain(|&id| id != edge_id);
                    if outgoing_list.is_empty() {
                        self.outgoing.remove(&edge.from);
                    }
                }

                // Remove from incoming list of the 'to' node
                if let Some(incoming_list) = self.incoming.get_mut(&edge.to) {
                    incoming_list.retain(|&id| id != edge_id);
                    if incoming_list.is_empty() {
                        self.incoming.remove(&edge.to);
                    }
                }
            }
        }

        // 4. Finally, remove any empty adjacency lists for the removed node
        self.incoming.remove(&root);
        self.outgoing.remove(&root);

        Ok(())
    }

    /// Get all of the Syntax Children of a node, ordered by their SyntaxSibling edges
    /// (i.e. in the order they appear in the source code)
    pub fn ordered_syntax_children(&self, root: NodeId) -> Vec<NodeId> {
        // Guard against no edges
        let outgoing_edges = self.get_outgoing_edges(root);
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
            let child_outgoing = self.get_outgoing_edges(child);
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

    /// Compare two CPGs for semantic equality
    /// Returns a detailed comparison result indicating structural differences and function-level mismatches
    pub fn compare(&self, other: &Cpg) -> Result<DetailedComparisonResult, CpgError> {
        debug!(
            "Comparing CPGs: left root = {:?}, right root = {:?}",
            self.get_root(),
            other.get_root()
        );

        let left_root = self.get_root();
        let right_root = other.get_root();

        match (left_root, right_root) {
            (None, None) => Ok(DetailedComparisonResult::Equivalent),
            (None, Some(_)) => Ok(DetailedComparisonResult::StructuralMismatch {
                only_in_left: vec![],
                only_in_right: vec!["root".to_string()],
                function_mismatches: vec![],
            }),
            (Some(_), None) => Ok(DetailedComparisonResult::StructuralMismatch {
                only_in_left: vec!["root".to_string()],
                only_in_right: vec![],
                function_mismatches: vec![],
            }),
            (Some(l_root), Some(r_root)) => {
                let l_node = self.get_node_by_id(&l_root).ok_or_else(|| {
                    CpgError::MissingField(format!(
                        "Node with id {:?} not found in left CPG",
                        l_root
                    ))
                })?;
                let r_node = other.get_node_by_id(&r_root).ok_or_else(|| {
                    CpgError::MissingField(format!(
                        "Node with id {:?} not found in right CPG",
                        r_root
                    ))
                })?;

                // Check if both roots are TranslationUnit nodes
                if l_node.type_ != NodeType::TranslationUnit
                    || r_node.type_ != NodeType::TranslationUnit
                {
                    // If roots aren't TranslationUnits, fall back to subtree comparison
                    let mut mismatches = Vec::new();
                    let mut visited = std::collections::HashSet::new();
                    self.compare_subtrees(other, &mut mismatches, l_root, r_root, &mut visited)?;
                    if mismatches.is_empty() {
                        return Ok(DetailedComparisonResult::Equivalent);
                    } else {
                        return Ok(DetailedComparisonResult::StructuralMismatch {
                            only_in_left: vec![],
                            only_in_right: vec![],
                            function_mismatches: vec![FunctionComparisonResult::Mismatch {
                                function_name: "root".to_string(),
                                details: format!(
                                    "Root nodes differ [Root not TranslationUnit (left={},right={})]: {}",
                                    l_node.type_ != NodeType::TranslationUnit,
                                    r_node.type_ != NodeType::TranslationUnit,
                                    mismatches
                                        .iter()
                                        .map(|(ol, or)| {
                                            format!(
                                                "Left: {:?}, Right: {:?}",
                                                ol.and_then(|l| self.nodes.get(l).and_then(|n| {
                                                    Some((
                                                        l,
                                                        n.type_.clone(),
                                                        self.spatial_index.get_range_from_node(&l),
                                                    ))
                                                })),
                                                or.and_then(|r| other.nodes.get(r).and_then(|n| {
                                                    Some((
                                                        r,
                                                        n.type_.clone(),
                                                        other.spatial_index.get_range_from_node(&r),
                                                    ))
                                                }))
                                            )
                                        })
                                        .collect::<Vec<_>>()
                                        .join(", ")
                                ),
                            }],
                        });
                    }
                }

                // Compare top-level structure
                let left_functions = self.get_top_level_functions(l_root)?;
                let right_functions = other.get_top_level_functions(r_root)?;

                let mut only_in_left = Vec::new();
                let mut only_in_right = Vec::new();
                let mut function_mismatches = Vec::new();

                // Find functions only in left CPG
                for (name, _) in &left_functions {
                    if !right_functions.contains_key(name) {
                        only_in_left.push(name.clone());
                    }
                }

                // Find functions only in right CPG
                for (name, _) in &right_functions {
                    if !left_functions.contains_key(name) {
                        only_in_right.push(name.clone());
                    }
                }

                // Compare functions present in both CPGs
                for (name, left_func_id) in &left_functions {
                    if let Some(right_func_id) = right_functions.get(name) {
                        let mut mismatches = Vec::new();
                        let mut visited = std::collections::HashSet::new();
                        self.compare_subtrees(
                            other,
                            &mut mismatches,
                            *left_func_id,
                            *right_func_id,
                            &mut visited,
                        )?;

                        if !mismatches.is_empty() {
                            function_mismatches.push(FunctionComparisonResult::Mismatch {
                                function_name: name.clone(),
                                details: format!(
                                    "Function {} has structural differences: {}",
                                    name,
                                    mismatches
                                        .iter()
                                        .map(|(ol, or)| {
                                            format!(
                                                "Left: {:?}, Right: {:?}",
                                                ol.and_then(|l| self.nodes.get(l).and_then(|n| {
                                                    Some((
                                                        l,
                                                        n.type_.clone(),
                                                        self.spatial_index.get_range_from_node(&l),
                                                    ))
                                                })),
                                                or.and_then(|r| other.nodes.get(r).and_then(|n| {
                                                    Some((
                                                        r,
                                                        n.type_.clone(),
                                                        other.spatial_index.get_range_from_node(&r),
                                                    ))
                                                }))
                                            )
                                        })
                                        .collect::<Vec<_>>()
                                        .join(", ")
                                ),
                            });
                        }
                    }
                }

                // Check if there are any differences
                if only_in_left.is_empty()
                    && only_in_right.is_empty()
                    && function_mismatches.is_empty()
                {
                    Ok(DetailedComparisonResult::Equivalent)
                } else {
                    Ok(DetailedComparisonResult::StructuralMismatch {
                        only_in_left,
                        only_in_right,
                        function_mismatches,
                    })
                }
            }
        }
    }

    /// Get all top-level function definitions from a TranslationUnit
    pub fn get_top_level_functions(
        &self,
        root: NodeId,
    ) -> Result<HashMap<String, NodeId>, CpgError> {
        let mut functions = HashMap::new();

        // Get all SyntaxChild edges from the root
        let child_edges = self.get_outgoing_edges(root);

        debug!(
            "Looking for functions in {} child edges from root {:?}",
            child_edges.len(),
            root
        );

        for edge in child_edges {
            if edge.type_ == EdgeType::SyntaxChild {
                let node = self.get_node_by_id(&edge.to).ok_or_else(|| {
                    CpgError::MissingField(format!("Child node with id {:?} not found", edge.to))
                })?;

                debug!(
                    "Child node type: {:?}, properties: {:?}",
                    node.type_, node.properties
                );

                // Check if this child is a Function node
                match node.type_ {
                    NodeType::Function { .. } => {
                        // Try to get the function name from properties
                        let name = node
                            .properties
                            .get("name")
                            .cloned()
                            .unwrap_or_else(|| format!("unnamed_function_{:?}", edge.to));
                        debug!("Found function with name: {}", name);
                        functions.insert(name, edge.to);
                    }
                    _ => {}
                }
            }
        }

        debug!(
            "Found {} functions: {:?}",
            functions.len(),
            functions.keys().collect::<Vec<_>>()
        );
        Ok(functions)
    }

    /// Compare two subtrees, updating a list of the NodeIds of the sub-subtrees that are mismatched
    fn compare_subtrees(
        &self,
        other: &Cpg,
        mismatches: &mut Vec<(Option<NodeId>, Option<NodeId>)>,
        l_root: NodeId,
        r_root: NodeId,
        visited: &mut HashSet<(NodeId, NodeId)>,
    ) -> Result<(), CpgError> {
        // Avoid re-comparing the same pair of nodes
        if !visited.insert((l_root, r_root)) {
            return Ok(());
        }

        let l_node = self.get_node_by_id(&l_root).ok_or_else(|| {
            CpgError::MissingField(format!("Node with id {:?} not found in left CPG", l_root))
        })?;

        let r_node = other.get_node_by_id(&r_root).ok_or_else(|| {
            CpgError::MissingField(format!("Node with id {:?} not found in right CPG", r_root))
        })?;

        if l_node != r_node {
            mismatches.push((Some(l_root), Some(r_root)));
            return Ok(());
        }

        let l_edges = self.get_outgoing_edges(l_root);
        let r_edges = other.get_outgoing_edges(r_root);

        let mut grouped_left: HashMap<(_, Vec<(_, _)>), Vec<_>> = HashMap::new();
        let mut grouped_right: HashMap<(_, Vec<(_, _)>), Vec<_>> = HashMap::new();

        for e in l_edges.iter() {
            grouped_left
                .entry((&e.type_, to_sorted_vec(&e.properties)))
                .or_default()
                .push(e);
        }
        for e in r_edges.iter() {
            grouped_right
                .entry((&e.type_, to_sorted_vec(&e.properties)))
                .or_default()
                .push(e);
        }

        for ((edge_type, props), left_group) in &grouped_left {
            let right_group = grouped_right.get(&(*edge_type, props.clone()));
            match right_group {
                Some(rg) => {
                    if **edge_type == EdgeType::SyntaxChild {
                        let ordered_left = self.ordered_syntax_children(l_root);
                        let ordered_right = other.ordered_syntax_children(r_root);

                        if ordered_left.len() != ordered_right.len() {
                            mismatches.push((Some(l_root), Some(r_root)));
                            return Ok(());
                        }

                        for (lc, rc) in ordered_left.iter().zip(ordered_right.iter()) {
                            self.compare_subtrees(other, mismatches, *lc, *rc, visited)?; // Pass visited
                        }
                    } else {
                        if left_group.len() != rg.len() {
                            mismatches.push((Some(l_root), Some(r_root)));
                            return Ok(());
                        }

                        for (l_edge, r_edge) in left_group.iter().zip(rg.iter()) {
                            self.compare_subtrees(
                                other, mismatches, l_edge.to, r_edge.to, visited,
                            )?; // Pass visited
                        }
                    }
                }
                None => {
                    mismatches.push((Some(l_root), Some(r_root)));
                    return Ok(());
                }
            }
        }

        Ok(())
    }

    pub fn emit_dot(&self) -> String {
        let mut dot = String::new();
        dot.push_str("digraph CPG {\n");
        dot.push_str("  rankdir=TB;\n");
        dot.push_str("  node [shape=box];\n");

        let mut visited = HashSet::new();

        fn emit_edge(dot: &mut String, edge: &Edge) {
            let from = format!("{:?}", edge.from)
                .replace("NodeId(", "")
                .replace(")", "");
            let to = format!("{:?}", edge.to)
                .replace("NodeId(", "")
                .replace(")", "");

            let col = match edge.type_ {
                EdgeType::Unknown => "black",
                EdgeType::SyntaxChild => "blue",
                EdgeType::SyntaxSibling => "green",
                EdgeType::ControlFlowEpsilon => "red",
                EdgeType::ControlFlowTrue => "orange",
                EdgeType::ControlFlowFalse => "purple",
                EdgeType::PDControlTrue => "cyan",
                EdgeType::PDControlFalse => "magenta",
                EdgeType::PDData(_) => "brown",
                EdgeType::Listener(_) => "gray",
            };

            dot.push_str(&format!(
                "  {:?} -> {:?} [label=\"{}\", color=\"{}\"];\n",
                from,
                to,
                edge.type_
                    .to_string()
                    .replace("EdgeType::", "")
                    .replace("_", " "),
                col
            ));
        }

        fn emit_node(dot: &mut String, cpg: &Cpg, node_id: NodeId, visited: &mut HashSet<NodeId>) {
            if !visited.insert(node_id) {
                return; // Already visited this node
            }

            let node = cpg.get_node_by_id(&node_id).expect("Node should exist");
            let id_s = format!("{:?}", node_id)
                .replace("NodeId(", "")
                .replace(")", "");

            let pos = cpg
                .spatial_index
                .get_range_from_node(&node_id)
                .map_or("unknown".to_string(), |(start, end)| {
                    format!("{}-{}", start, end)
                });

            dot.push_str(&format!(
                "  {:?} [label=\"{} {} {}\" color={}];\n",
                id_s,
                node.type_
                    .to_string()
                    .replace("NodeType::", "")
                    .replace("_", " "),
                pos,
                node.properties
                    .get("raw_kind")
                    .cloned()
                    .unwrap_or_else(|| "unnamed".to_string()),
                match node.type_ {
                    NodeType::Comment | NodeType::LanguageImplementation(_) => "lightgray",
                    _ => "black",
                }
            ));

            for edge in cpg.get_outgoing_edges(node_id) {
                emit_node(dot, cpg, edge.to, visited);
            }
        }

        // Recursive walk emitting nodes
        if let Some(root) = self.get_root() {
            emit_node(&mut dot, self, root, &mut visited);
        } else {
            dot.push_str("  // No root node set\n");
        }

        // Emit edges
        for edge in self.edges.values() {
            emit_edge(&mut dot, edge);
        }

        dot.push_str("}\n");
        dot
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

// --- Tests --- //

#[cfg(test)]
mod tests {
    use super::*;

    // --- Helper functions for tests --- //

    fn create_test_cpg() -> Cpg {
        Cpg::new("C".parse().expect("Failed to parse language"), Vec::new())
    }

    fn create_test_node(node_type: NodeType) -> Node {
        Node {
            type_: node_type.clone(),
            properties: {
                let mut prop = HashMap::new();
                match &node_type {
                    NodeType::Function { name, .. } => {
                        if let Some(n) = name {
                            prop.insert("name".to_string(), n.clone());
                        }
                    }
                    _ => {}
                }
                prop
            },
        }
    }

    // --- Basic functionality tests --- //

    #[test]
    fn test_cpg_creation() {
        let cpg = Cpg::new("C".parse().expect("Failed to parse language"), Vec::new());
        assert!(cpg.nodes.is_empty());
        assert!(cpg.edges.is_empty());
        assert!(cpg.get_root().is_none());
    }

    #[test]
    fn test_add_node() {
        let mut cpg = create_test_cpg();
        let node = create_test_node(NodeType::TranslationUnit);
        let node_id = cpg.add_node(node.clone(), 0, 10);

        assert_eq!(cpg.get_node_by_id(&node_id), Some(&node));
        assert_eq!(cpg.get_root(), Some(node_id));
        assert_eq!(
            cpg.spatial_index.get_range_from_node(&node_id),
            Some((0, 10))
        );
    }

    #[test]
    fn test_add_edge() {
        let mut cpg = create_test_cpg();
        let node_id1 = cpg.add_node(
            create_test_node(NodeType::Function {
                name_traversal: desc_trav![],
                name: Some("text_func".to_string()),
            }),
            0,
            1,
        );
        let node_id2 = cpg.add_node(create_test_node(NodeType::Identifier), 1, 2);

        let edge = Edge {
            from: node_id1,
            to: node_id2,
            type_: EdgeType::SyntaxChild,
            properties: HashMap::new(),
        };

        let edge_id = cpg.add_edge(edge.clone());
        assert_eq!(cpg.edges.get(edge_id), Some(&edge));

        // Test adjacency lists are updated
        let outgoing = cpg.get_outgoing_edges(node_id1);
        assert_eq!(outgoing.len(), 1);
        assert_eq!(outgoing[0], &edge);

        let incoming = cpg.get_incoming_edges(node_id2);
        assert_eq!(incoming.len(), 1);
        assert_eq!(incoming[0], &edge);
    }

    #[test]
    fn test_set_root_override() {
        let mut cpg = create_test_cpg();
        let node1 = cpg.add_node(
            create_test_node(NodeType::Function {
                name_traversal: desc_trav![],
                name: Some("test".to_string()),
            }),
            0,
            1,
        );
        let node2 = cpg.add_node(create_test_node(NodeType::TranslationUnit), 1, 2);

        assert_eq!(cpg.get_root(), Some(node1)); // First node becomes root

        cpg.set_root(node2);
        assert_eq!(cpg.get_root(), Some(node2)); // Root can be overridden
    }

    // --- Spatial index tests --- //

    #[test]
    fn test_spatial_index_basic() {
        let mut cpg = create_test_cpg();
        let node_id1 = cpg.add_node(
            create_test_node(NodeType::Function {
                name_traversal: desc_trav![],
                name: Some("main".to_string()),
            }),
            0,
            10,
        );
        let node_id2 = cpg.add_node(create_test_node(NodeType::Identifier), 5, 15);
        let _node_id3 = cpg.add_node(create_test_node(NodeType::Identifier), 12, 20);

        let overlapping = cpg.spatial_index.lookup_nodes_from_range(8, 12);
        assert_eq!(overlapping.len(), 2);
        assert!(overlapping.contains(&&node_id1));
        assert!(overlapping.contains(&&node_id2));

        let non_overlapping = cpg.spatial_index.lookup_nodes_from_range(21, 25);
        assert!(non_overlapping.is_empty());

        cpg.spatial_index.remove_by_node(&node_id2);
        let after_removal = cpg.spatial_index.lookup_nodes_from_range(8, 12);
        assert_eq!(after_removal.len(), 1);
        assert!(after_removal.contains(&&node_id1));
    }

    #[test]
    fn test_spatial_index_edge_cases() {
        let mut cpg = create_test_cpg();

        // Test zero-width ranges
        let _node_id = cpg.add_node(create_test_node(NodeType::Identifier), 5, 5);
        let overlapping = cpg.spatial_index.lookup_nodes_from_range(5, 5);
        assert!(overlapping.is_empty()); // Zero-width ranges don't overlap

        // Test exact boundaries
        let node_id2 = cpg.add_node(
            create_test_node(NodeType::Function {
                name_traversal: desc_trav![],
                name: Some("main".to_string()),
            }),
            0,
            10,
        );
        let exact_match = cpg.spatial_index.lookup_nodes_from_range(0, 10);
        // Note: The spatial index includes the first node added (root), so count should be 2
        assert_eq!(exact_match.len(), 2);
        assert!(exact_match.contains(&&node_id2));

        // Test adjacent ranges
        let _node_id3 = cpg.add_node(create_test_node(NodeType::Statement), 10, 20);
        let adjacent = cpg.spatial_index.lookup_nodes_from_range(10, 10);
        assert!(adjacent.is_empty()); // Adjacent ranges shouldn't overlap
    }

    #[test]
    fn test_get_smallest_node_containing_range() {
        let mut cpg = create_test_cpg();
        let large_node = cpg.add_node(
            create_test_node(NodeType::Function {
                name_traversal: desc_trav![],
                name: Some("main".to_string()),
            }),
            0,
            100,
        );
        let medium_node = cpg.add_node(create_test_node(NodeType::Block), 10, 50);
        let small_node = cpg.add_node(create_test_node(NodeType::Identifier), 20, 30);

        let result = cpg.get_smallest_node_id_containing_range(25, 26);
        assert_eq!(result, Some(small_node));

        let result2 = cpg.get_smallest_node_id_containing_range(15, 45);
        assert_eq!(result2, Some(medium_node));

        let result3 = cpg.get_smallest_node_id_containing_range(5, 95);
        assert_eq!(result3, Some(large_node));

        let result4 = cpg.get_smallest_node_id_containing_range(200, 300);
        assert_eq!(result4, None);
    }

    // --- Edge query tests --- //

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

    // --- Subtree removal tests --- //

    #[test]
    fn test_remove_subtree_simple() {
        let mut cpg = create_test_cpg();
        let root = cpg.add_node(
            create_test_node(NodeType::Function {
                name_traversal: desc_trav![],
                name: Some("main".to_string()),
            }),
            0,
            10,
        );
        let child1 = cpg.add_node(create_test_node(NodeType::Statement), 1, 5);
        let child2 = cpg.add_node(create_test_node(NodeType::Identifier), 6, 9);

        cpg.add_edge(Edge {
            from: root,
            to: child1,
            type_: EdgeType::SyntaxChild,
            properties: HashMap::new(),
        });
        cpg.add_edge(Edge {
            from: root,
            to: child2,
            type_: EdgeType::SyntaxChild,
            properties: HashMap::new(),
        });

        let initial_node_count = cpg.nodes.len();
        let initial_edge_count = cpg.edges.len();

        cpg.remove_subtree(child1)
            .expect("Failed to remove subtree");

        // Child1 should be removed
        assert!(cpg.get_node_by_id(&child1).is_none());
        assert_eq!(cpg.nodes.len(), initial_node_count - 1);
        assert_eq!(cpg.edges.len(), initial_edge_count - 1);

        // Root and child2 should remain
        assert!(cpg.get_node_by_id(&root).is_some());
        assert!(cpg.get_node_by_id(&child2).is_some());
    }

    #[test]
    fn test_remove_subtree_recursive() {
        let mut cpg = create_test_cpg();
        let root = cpg.add_node(
            create_test_node(NodeType::Function {
                name_traversal: desc_trav![],
                name: Some("main".to_string()),
            }),
            0,
            20,
        );
        let child1 = cpg.add_node(create_test_node(NodeType::Block), 1, 10);
        let grandchild = cpg.add_node(create_test_node(NodeType::Statement), 2, 8);
        let child2 = cpg.add_node(create_test_node(NodeType::Identifier), 11, 19);

        cpg.add_edge(Edge {
            from: root,
            to: child1,
            type_: EdgeType::SyntaxChild,
            properties: HashMap::new(),
        });
        cpg.add_edge(Edge {
            from: child1,
            to: grandchild,
            type_: EdgeType::SyntaxChild,
            properties: HashMap::new(),
        });
        cpg.add_edge(Edge {
            from: root,
            to: child2,
            type_: EdgeType::SyntaxChild,
            properties: HashMap::new(),
        });

        cpg.remove_subtree(child1)
            .expect("Failed to remove subtree");

        // Both child1 and grandchild should be removed
        assert!(cpg.get_node_by_id(&child1).is_none());
        assert!(cpg.get_node_by_id(&grandchild).is_none());

        // Root and child2 should remain
        assert!(cpg.get_node_by_id(&root).is_some());
        assert!(cpg.get_node_by_id(&child2).is_some());

        // No dangling edges should remain
        assert!(cpg.get_outgoing_edges(child1).is_empty());
        assert!(cpg.get_incoming_edges(child1).is_empty());
    }

    #[test]
    fn test_remove_subtree_edge_cleanup() {
        let mut cpg = create_test_cpg();
        let node1 = cpg.add_node(
            create_test_node(NodeType::Function {
                name_traversal: desc_trav![],
                name: Some("main".to_string()),
            }),
            0,
            5,
        );
        let node2 = cpg.add_node(create_test_node(NodeType::Statement), 6, 10);
        let node3 = cpg.add_node(create_test_node(NodeType::Identifier), 11, 15);

        // Create a complex edge structure
        let edge1 = cpg.add_edge(Edge {
            from: node1,
            to: node2,
            type_: EdgeType::SyntaxChild,
            properties: HashMap::new(),
        });
        let edge2 = cpg.add_edge(Edge {
            from: node2,
            to: node3,
            type_: EdgeType::ControlFlowTrue,
            properties: HashMap::new(),
        });
        let edge3 = cpg.add_edge(Edge {
            from: node1,
            to: node3,
            type_: EdgeType::PDData("x".to_string()),
            properties: HashMap::new(),
        });

        // Remove node2, which should clean up associated edges
        cpg.remove_subtree(node2).expect("Failed to remove subtree");

        // Check that edges involving node2 are removed
        assert!(cpg.edges.get(edge1).is_none());
        assert!(cpg.edges.get(edge2).is_none());
        assert!(cpg.edges.get(edge3).is_some()); // This edge should remain

        // Check adjacency lists are updated
        assert!(cpg.get_outgoing_edges(node2).is_empty());
        assert!(cpg.get_incoming_edges(node2).is_empty());

        // Node1 should no longer have edge to node2
        let node1_outgoing = cpg.get_outgoing_edges(node1);
        assert_eq!(node1_outgoing.len(), 1);
        assert_eq!(node1_outgoing[0].to, node3);
    }

    // --- Ordered syntax children tests --- //

    #[test]
    fn test_ordered_syntax_children() {
        let mut cpg = create_test_cpg();
        let parent = cpg.add_node(
            create_test_node(NodeType::Function {
                name_traversal: desc_trav![],
                name: Some("main".to_string()),
            }),
            0,
            30,
        );
        let child1 = cpg.add_node(create_test_node(NodeType::Statement), 1, 10);
        let child2 = cpg.add_node(create_test_node(NodeType::Statement), 11, 20);
        let child3 = cpg.add_node(create_test_node(NodeType::Statement), 21, 29);

        // Add syntax child edges
        cpg.add_edge(Edge {
            from: parent,
            to: child1,
            type_: EdgeType::SyntaxChild,
            properties: HashMap::new(),
        });
        cpg.add_edge(Edge {
            from: parent,
            to: child2,
            type_: EdgeType::SyntaxChild,
            properties: HashMap::new(),
        });
        cpg.add_edge(Edge {
            from: parent,
            to: child3,
            type_: EdgeType::SyntaxChild,
            properties: HashMap::new(),
        });

        // Add sibling edges to establish order
        cpg.add_edge(Edge {
            from: child1,
            to: child2,
            type_: EdgeType::SyntaxSibling,
            properties: HashMap::new(),
        });
        cpg.add_edge(Edge {
            from: child2,
            to: child3,
            type_: EdgeType::SyntaxSibling,
            properties: HashMap::new(),
        });

        let ordered = cpg.ordered_syntax_children(parent);
        assert_eq!(ordered, vec![child1, child2, child3]);
    }

    #[test]
    fn test_ordered_syntax_children_empty() {
        let mut cpg = create_test_cpg();
        let parent = cpg.add_node(
            create_test_node(NodeType::Function {
                name_traversal: desc_trav![],
                name: Some("main".to_string()),
            }),
            0,
            10,
        );

        let ordered = cpg.ordered_syntax_children(parent);
        assert!(ordered.is_empty());
    }

    // --- CPG comparison tests --- //

    #[test]
    fn test_compare_equivalent_cpgs() {
        let mut cpg1 = create_test_cpg();
        let mut cpg2 = create_test_cpg();

        // Create identical structures
        for cpg in [&mut cpg1, &mut cpg2] {
            let root = cpg.add_node(create_test_node(NodeType::TranslationUnit), 0, 20);
            let func = cpg.add_node(
                create_test_node(NodeType::Function {
                    name_traversal: desc_trav![],
                    name: Some("main".to_string()),
                }),
                1,
                19,
            );

            cpg.add_edge(Edge {
                from: root,
                to: func,
                type_: EdgeType::SyntaxChild,
                properties: HashMap::new(),
            });
        }

        let result = cpg1.compare(&cpg2).expect("Comparison failed");
        assert!(matches!(result, DetailedComparisonResult::Equivalent));
    }

    #[test]
    fn test_compare_different_functions() {
        let mut cpg1 = create_test_cpg();
        let mut cpg2 = create_test_cpg();

        // CPG1 has function "main"
        let root1 = cpg1.add_node(create_test_node(NodeType::TranslationUnit), 0, 20);
        let func1 = cpg1.add_node(
            create_test_node(NodeType::Function {
                name_traversal: desc_trav![],
                name: Some("main".to_string()),
            }),
            1,
            19,
        );
        cpg1.add_edge(Edge {
            from: root1,
            to: func1,
            type_: EdgeType::SyntaxChild,
            properties: HashMap::new(),
        });

        // CPG2 has function "test"
        let root2 = cpg2.add_node(create_test_node(NodeType::TranslationUnit), 0, 20);
        let func2 = cpg2.add_node(
            create_test_node(NodeType::Function {
                name_traversal: desc_trav![],
                name: Some("test".to_string()),
            }),
            1,
            19,
        );
        cpg2.add_edge(Edge {
            from: root2,
            to: func2,
            type_: EdgeType::SyntaxChild,
            properties: HashMap::new(),
        });

        let result = cpg1.compare(&cpg2).expect("Comparison failed");
        match result {
            DetailedComparisonResult::StructuralMismatch {
                only_in_left,
                only_in_right,
                ..
            } => {
                assert!(only_in_left.contains(&"main".to_string()));
                assert!(only_in_right.contains(&"test".to_string()));
            }
            _ => panic!("Expected structural mismatch"),
        }
    }

    #[test]
    fn test_compare_no_roots() {
        let cpg1 = create_test_cpg();
        let cpg2 = create_test_cpg();

        let result = cpg1.compare(&cpg2).expect("Comparison failed");
        assert!(matches!(result, DetailedComparisonResult::Equivalent));
    }

    #[test]
    fn test_compare_one_empty() {
        let cpg1 = create_test_cpg();
        let mut cpg2 = create_test_cpg();
        cpg2.add_node(create_test_node(NodeType::TranslationUnit), 0, 10);

        let result = cpg1.compare(&cpg2).expect("Comparison failed");
        match result {
            DetailedComparisonResult::StructuralMismatch { only_in_right, .. } => {
                assert!(only_in_right.contains(&"root".to_string()));
            }
            _ => panic!("Expected structural mismatch"),
        }
    }

    // --- Node offset query tests --- //

    #[test]
    fn test_get_node_by_offsets() {
        let mut cpg = create_test_cpg();
        let node1 = cpg.add_node(
            create_test_node(NodeType::Function {
                name_traversal: desc_trav![],
                name: Some("main".to_string()),
            }),
            0,
            10,
        );
        let node2 = cpg.add_node(create_test_node(NodeType::Identifier), 5, 15);

        let nodes = cpg.get_node_by_offsets(8, 12);
        assert_eq!(nodes.len(), 2);

        let node_ids = cpg.get_node_ids_by_offsets(8, 12);
        assert_eq!(node_ids.len(), 2);
        assert!(node_ids.contains(&node1));
        assert!(node_ids.contains(&node2));
    }

    // --- Error handling tests --- //

    #[test]
    fn test_remove_nonexistent_subtree() {
        let mut cpg = create_test_cpg();
        let node = cpg.add_node(
            create_test_node(NodeType::Function {
                name_traversal: desc_trav![],
                name: Some("main".to_string()),
            }),
            0,
            10,
        );
        cpg.nodes.remove(node); // Manually remove to create invalid state

        // This should handle the case gracefully
        let result = cpg.remove_subtree(node);
        assert!(result.is_ok()); // Should not panic
    }
}
