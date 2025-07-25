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

// fn to_sorted_vec(map: &HashMap<String, String>) -> Vec<(&String, &String)> {
//     let mut vec: Vec<_> = map.iter().collect();
//     vec.sort();
//     vec
// }

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

// --- The graph structure for the CPG --- //

#[derive(Debug, Clone, Display, PartialEq, Eq, Hash)]
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

    // Control flow constructs
    If,         // If statement
    While,      // While loop
    For,        // For loop
    Condition,  // Condition expression (used in if/while/for)
    ThenBranch, // Then branch of an if statement
    ElseBranch, // Else branch of an if statement
    LoopBody,   // Body of a loop
    LoopInit,   // Initialization part of a for loop
    LoopUpdate, // Update part of a for loop

    // The weeds (should these be subtypes of statement, expression, etc. or their own types?)
    Call,   // A function call expression
    Return, // A return statement in a function
    Block,  // Compound statement, e.g., a block of code enclosed in braces
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    pub type_: NodeType,
    pub properties: HashMap<String, String>, // As little as possible should be stored here
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
    pub fn new(lang: RegisteredLanguage) -> Self {
        Cpg {
            root: None,
            nodes: SlotMap::with_key(),
            edges: SlotMap::with_key(),
            incoming: HashMap::new(),
            outgoing: HashMap::new(),
            spatial_index: SpatialIndex::new(),
            language: lang,
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
            .min_by_key(|id| {
                let range = self
                    .spatial_index
                    .get_range_from_node(id)
                    .expect("NodeId should have a range");
                range.1 - range.0 // Return the size of the node
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
                    (edit.old_end, edit.old_end)
                );
            }
        }

        debug!("Rehydrating dirty nodes: {:?}", dirty_nodes);
        let mut rehydrated_nodes = Vec::new();
        for (id, pos) in dirty_nodes {
            let new_node = self.rehydrate(id, pos, new_tree);
            match new_node {
                Ok(new_id) => {
                    debug!("Rehydrated node {:?} to new id {:?}", id, new_id);
                    rehydrated_nodes.push(new_id);
                }
                Err(e) => {
                    debug!("Failed to rehydrate node {:?}: {}", id, e);
                }
            }
        }

        debug!(
            "Computing control flow for {} rehydrated nodes",
            rehydrated_nodes.len()
        );
        for new_node in rehydrated_nodes.clone() {
            cf_pass(self, new_node)
                .map_err(|e| {
                    debug!(
                        "Failed to recompute control flow for node {:?}: {}",
                        new_node, e
                    )
                })
                .ok();
        }

        debug!(
            "Computing program dependence for {} rehydrated nodes",
            rehydrated_nodes.len()
        );
        for new_node in rehydrated_nodes {
            data_dep_pass(self, new_node)
                .map_err(|e| {
                    debug!(
                        "Failed to recompute data dependence for node {:?}: {}",
                        new_node, e
                    )
                })
                .ok();
        }

        debug!("Incremental update complete");
    }

    fn rehydrate(
        &mut self,
        id: NodeId,
        pos: (usize, usize, usize, usize),
        new_tree: &tree_sitter::Tree,
    ) -> Result<NodeId, CpgError> {
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
        } else {
            debug!(
                "No parent edge found for node {:?}, assuming it is a root node",
                id
            );
            self.set_root(new_subtree_root);
        }

        // Remove the old subtree from the CPG
        self.remove_subtree(id).map_err(|e| {
            CpgError::ConversionError(format!("Failed to remove old subtree: {}", e))
        })?;

        Ok(new_subtree_root)
    }

    /// Recursively removes a subtree from the CPG by its root node ID
    /// This function now properly cleans up edges associated with the removed nodes.
    fn remove_subtree(&mut self, root: NodeId) -> Result<(), CpgError> {
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
        // Collect edge IDs to avoid borrowing issues while modifying the maps
        let incoming_edge_ids: Vec<EdgeId> = self.incoming.remove(&root).unwrap_or_default();
        let outgoing_edge_ids: Vec<EdgeId> = self.outgoing.remove(&root).unwrap_or_default();

        // Remove the actual Edge structs from the main edges SlotMap
        // And also remove them from the counterpart adjacency lists
        for edge_id in incoming_edge_ids {
            if let Some(edge) = self.edges.remove(edge_id) {
                // Remove this edge from the outgoing list of its 'from' node
                if let Some(outgoing_list) = self.outgoing.get_mut(&edge.from) {
                    outgoing_list.retain(|&id| id != edge_id);
                    // If the list becomes empty, we could optionally remove the key,
                    // but keeping it is usually fine.
                    // if outgoing_list.is_empty() { self.outgoing.remove(&edge.from); }
                }
            }
        }

        for edge_id in outgoing_edge_ids {
            if let Some(edge) = self.edges.remove(edge_id) {
                // Remove this edge from the incoming list of its 'to' node
                if let Some(incoming_list) = self.incoming.get_mut(&edge.to) {
                    incoming_list.retain(|&id| id != edge_id);
                    // if incoming_list.is_empty() { self.incoming.remove(&edge.to); }
                }
            }
        }

        Ok(())
    }

    /// Get all of the Syntax Children of a node, ordered by their SyntaxSibling edges
    /// (i.e. in the order they appear in the source code)
    pub fn ordered_syntax_children(&self, root: NodeId) -> Vec<NodeId> {
        let edges = self.get_outgoing_edges(root);
        let child_targets: Vec<NodeId> = edges
            .iter()
            .filter(|e| e.type_ == EdgeType::SyntaxChild)
            .map(|e| e.to)
            .collect();

        let sibling_map: HashMap<NodeId, NodeId> = self
            .edges
            .iter()
            .filter(|(_, e)| e.type_ == EdgeType::SyntaxSibling)
            .map(|(_, e)| (e.from, e.to))
            .collect();

        let all_targets: std::collections::HashSet<_> = sibling_map.values().copied().collect();
        let start = child_targets
            .iter()
            .find(|n| !all_targets.contains(n))
            .copied();

        let mut ordered = Vec::new();
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
                                details: "Root nodes differ".to_string(),
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
                                details: format!("Function {} has structural differences", name),
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
    fn get_top_level_functions(&self, root: NodeId) -> Result<HashMap<String, NodeId>, CpgError> {
        let mut functions = HashMap::new();

        // Get all SyntaxChild edges from the root
        let child_edges = self.get_outgoing_edges(root);

        for edge in child_edges {
            if edge.type_ == EdgeType::SyntaxChild {
                let node = self.get_node_by_id(&edge.to).ok_or_else(|| {
                    CpgError::MissingField(format!("Child node with id {:?} not found", edge.to))
                })?;

                // Check if this child is a Function node
                if node.type_ == NodeType::Function {
                    // Try to get the function name from properties
                    let name = node
                        .properties
                        .get("name")
                        .cloned()
                        .unwrap_or_else(|| format!("unnamed_function_{:?}", edge.to));
                    functions.insert(name, edge.to);
                }
            }
        }

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
    use crate::diff::incremental_parse;
    use crate::languages::RegisteredLanguage;
    use crate::resource::Resource;

    #[test]
    fn test_cpg_creation() {
        let cpg = Cpg::new("C".parse().expect("Failed to parse language"));
        assert!(cpg.nodes.is_empty());
        assert!(cpg.edges.is_empty());
    }

    #[test]
    fn test_add_node() {
        let mut cpg = Cpg::new("C".parse().expect("Failed to parse language"));
        let node = Node {
            type_: NodeType::TranslationUnit,
            properties: HashMap::new(),
        };
        let node_id = cpg.add_node(node.clone(), 0, 10);
        assert_eq!(cpg.get_node_by_id(&node_id), Some(&node));
    }

    #[test]
    fn test_add_edge() {
        let mut cpg = Cpg::new("C".parse().expect("Failed to parse language"));
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
        let mut cpg = Cpg::new("C".parse().expect("Failed to parse language"));
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
        let mut cpg = Cpg::new("C".parse().expect("Failed to parse language"));
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
        let mut cpg = Cpg::new("C".parse().expect("Failed to parse language"));
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

        let overlapping = cpg.spatial_index.lookup_nodes_from_range(8, 12);
        assert_eq!(overlapping.len(), 2);
        assert!(overlapping.contains(&&node_id1));
        assert!(overlapping.contains(&&node_id2));

        let non_overlapping = cpg.spatial_index.lookup_nodes_from_range(21, 25);
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

        // Perform the incremental update
        cpg.incremental_update(edits, changed_ranges, &new_tree);

        // Compute the reference CPG
        let mut new_cpg = lang
            .cst_to_cpg(new_tree)
            .expect("Failed to convert new tree to CPG");

        // Check the difference between the two CPGs
        let diff = cpg.compare(&mut new_cpg).expect("Failed to compare CPGs");
        assert!(
            matches!(diff, DetailedComparisonResult::Equivalent),
            "CPGs should be semantically equivalent, but found differences: {:?}",
            diff
        );
    }
}
