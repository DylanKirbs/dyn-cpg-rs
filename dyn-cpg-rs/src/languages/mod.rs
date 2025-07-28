use std::collections::HashMap;

/// This module defines the `Language` trait and provides macros to register languages.
/// Each language should be defined in its own module and implement the `Language` trait, a macro has been provided to simplify this process.
use crate::cpg::{Cpg, DescendantTraversal, Edge, EdgeType, Node, NodeId, NodeType};

mod c;

use c::C;
use tracing::{debug, warn};

// --- The Language Trait and Construction Macros --- //

trait Language: Default + std::fmt::Debug {
    /// The display name of the language.
    const DISPLAY_NAME: &'static str;

    /// The variant names and extensions for the language.
    const VARIANT_NAMES: &'static [&'static str];

    /// Get a Tree-sitter parser for the language.
    /// Each call to this function returns a new parser instance.
    fn get_parser(&self) -> Result<tree_sitter::Parser, String>;

    /// Map a Tree-sitter node kind to a CPG node type.
    fn map_node_kind(&self, node_kind: &'static str) -> NodeType;
}

#[macro_export]
/// Macro to define a new language complying with the `Language` trait.
macro_rules! define_language {
    (
        $name:ident, [$($variant_names:expr),+], $lang:path, $map_kind:expr
    ) => {

            #[derive(Debug, Clone)]
            /// A struct representing the language, implementing the `Language` trait.
            pub struct $name;

            impl Default for $name {
                fn default() -> Self {
                    Self
                }
            }

            impl Language for $name {
                const DISPLAY_NAME: &'static str = stringify!($name);
                const VARIANT_NAMES: &'static [&'static str] = &[$($variant_names),+];

                fn get_parser(&self) -> Result<tree_sitter::Parser, String> {
                    let mut parser = tree_sitter::Parser::new();
                    parser.set_language(&($lang).into())
                        .map(|_| parser)
                        .map_err(|e| format!("Failed to set parser for {}: {}", stringify!($name), e))
                }

                fn map_node_kind(&self, node_kind: &'static str) -> NodeType {
                    $map_kind(self, node_kind)
                }
            }

    };
}

/// Macro to register multiple languages, creating an enum to handle them.
macro_rules! register_languages {
    (
        $($variant:ident),+
    ) => {

        #[derive(Debug, Clone)]
        /// An enum representing all registered languages and allowing for easy parsing and handling.
        pub enum RegisteredLanguage {
            $(
                $variant($variant),
            )+
        }

        impl std::str::FromStr for RegisteredLanguage {
            type Err = String;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                let s = s.to_lowercase();
                let candidates = vec![
                    $(
                        (&$variant::VARIANT_NAMES, RegisteredLanguage::$variant($variant::default())),
                    )+
                ];
                for (aliases, language) in candidates {
                    if aliases.contains(&s.as_str()) {
                        return Ok(language);
                    }
                }
                Err(format!("Unknown language: {}", s))
            }
        }

        impl RegisteredLanguage {
            pub fn get_display_name(&self) -> &'static str {
                match self {
                    $(RegisteredLanguage::$variant(_) => <$variant as Language>::DISPLAY_NAME,)+
                }
            }

            pub fn get_variant_names(&self) -> &'static [&'static str] {
                match self {
                    $(RegisteredLanguage::$variant(_) => <$variant as Language>::VARIANT_NAMES,)+
                }
            }

            pub fn get_parser(&self) -> Result<tree_sitter::Parser, String> {
                match self {
                    $(RegisteredLanguage::$variant(language) => language.get_parser(),)+
                }
            }

            pub fn map_node_kind(&self, node_kind: &'static str) -> NodeType {
                match self {
                    $(RegisteredLanguage::$variant(language) => language.map_node_kind(node_kind),)+
                }
            }

            pub fn cst_to_cpg(&self, tree: tree_sitter::Tree, source: Vec<u8>) -> Result<Cpg, String> {
                match self {
                    $(RegisteredLanguage::$variant(_) => cst_to_cpg(&self, tree, source),)+
                }
            }
        }
    };
}

// --- Language Definitions --- //

register_languages! { C }

// --- Generic Language Utilities --- //

pub fn cst_to_cpg(
    lang: &RegisteredLanguage,
    tree: tree_sitter::Tree,
    source: Vec<u8>,
) -> Result<Cpg, String> {
    debug!("Converting CST to CPG for {:?}", lang);

    let mut cpg = Cpg::new(lang.clone(), source);
    let mut cursor = tree.walk();

    translate(&mut cpg, &mut cursor).map_err(|e| {
        warn!("Failed to translate CST: {}", e);
        e
    })?;

    let root_opt = cpg.get_root();

    if root_opt.is_none() {
        return Err("CST translation resulted in an empty CPG".to_string());
    }

    let root = root_opt.expect("Root node must be present if it is not None");

    cf_pass(&mut cpg, root).map_err(|e| {
        warn!("Failed to compute control flow: {}", e);
        e
    })?;

    data_dep_pass(&mut cpg, root).map_err(|e| {
        warn!("Failed to compute data dependence: {}", e);
        e
    })?;

    Ok(cpg)
}

pub fn translate(cpg: &mut Cpg, cursor: &mut tree_sitter::TreeCursor) -> Result<NodeId, String> {
    let node = cursor.node();
    let source_len = cpg.get_source().len();

    // Validate node range is within source bounds
    if node.start_byte() > source_len || node.end_byte() > source_len {
        return Err(format!(
            "Node range ({}, {}) exceeds source length {}",
            node.start_byte(),
            node.end_byte(),
            source_len
        ));
    }

    let type_ = cpg.get_language().map_node_kind(node.kind());

    let mut cpg_node = Node {
        type_: type_.clone(),
        properties: HashMap::new(),
    };
    cpg_node
        .properties
        .insert("raw_kind".to_string(), node.kind().to_string());

    let mut fn_name = None;
    match type_ {
        NodeType::Function {
            name_traversal,
            name: None,
        } => {
            let id_node = name_traversal.get_ts_descendant(node);
            if let Some(id_node) = id_node {
                fn_name = id_node
                    .utf8_text(&cpg.get_source())
                    .map(|s| s.to_string())
                    .ok();
            }
        }
        NodeType::Function { name, .. } => {
            fn_name = name.clone();
        }
        _ => {}
    }
    if let Some(name) = fn_name {
        cpg_node.properties.insert("name".to_string(), name.clone());
        debug!(
            "Node found: {:?} with name: {}",
            node.kind(),
            cpg_node.properties["name"]
        );
    }

    let id = cpg.add_node(cpg_node, node.start_byte(), node.end_byte());

    if cursor.goto_first_child() {
        let mut left_child_id: Option<NodeId> = None;

        loop {
            let child_id = translate(cpg, cursor)?;

            // Edge from parent to child
            cpg.add_edge(Edge {
                from: id.clone(),
                to: child_id.clone(),
                type_: EdgeType::SyntaxChild,
                properties: HashMap::new(),
            });

            // Edge from left sibling to current
            if let Some(left_id) = &left_child_id {
                cpg.add_edge(Edge {
                    from: left_id.clone(),
                    to: child_id.clone(),
                    type_: EdgeType::SyntaxSibling,
                    properties: HashMap::new(),
                });
            }

            left_child_id = Some(child_id);

            if !cursor.goto_next_sibling() {
                break;
            }
        }

        cursor.goto_parent();
    }

    Ok(id)
}

// -- Helpers -- //

fn is_statement_node(node_type: &NodeType) -> bool {
    matches!(
        node_type,
        NodeType::Statement
            // | NodeType::Expression
            | NodeType::Call
            | NodeType::Block
            | NodeType::Conditional { .. }
            | NodeType::Loop { .. }
    )
}

fn find_child_by_type(cpg: &Cpg, children: &[NodeId], target_type: NodeType) -> Option<NodeId> {
    for &child in children {
        if let Some(node) = cpg.get_node_by_id(&child) {
            if node.type_ == target_type {
                return Some(child);
            }
        }
    }
    None
}

fn get_first_statement_in_block(cpg: &Cpg, block_node: NodeId) -> Option<NodeId> {
    let children = cpg.ordered_syntax_children(block_node);
    for child in children {
        if let Some(node) = cpg.get_node_by_id(&child) {
            if is_statement_node(&node.type_) {
                return Some(child);
            }
        }
    }
    None
}

fn get_last_statement_in_block(cpg: &Cpg, block_node: NodeId) -> Option<NodeId> {
    let children = cpg.ordered_syntax_children(block_node);
    for child in children.iter().rev() {
        if let Some(node) = cpg.get_node_by_id(child) {
            if is_statement_node(&node.type_) {
                return Some(*child);
            }
        }
    }
    None
}

fn add_control_flow_edge(
    cpg: &mut Cpg,
    from: NodeId,
    to: NodeId,
    edge_type: EdgeType,
) -> Result<(), String> {
    let existing_edges = cpg.get_outgoing_edges(from);
    for edge in existing_edges {
        if edge.to == to && edge.type_ == edge_type {
            debug!("Control flow edge already exists: {:?} -> {:?}", from, to);
            return Ok(());
        }
    }

    cpg.add_edge(Edge {
        from,
        to,
        type_: edge_type,
        properties: HashMap::new(),
    });

    Ok(())
}

// --- Control Flow & Data Dependence --- //

/// Idempotent computation of the control flow for a subtree in the CPG.
/// This pass assumes that the AST has been construed into a CPG.
pub fn cf_pass(cpg: &mut Cpg, subtree_root: NodeId) -> Result<(), String> {
    use std::collections::{HashSet, VecDeque};

    debug!(
        "Computing control flow for subtree root: {:?}",
        subtree_root
    );

    let mut queue = VecDeque::new();
    let mut visited = HashSet::new();
    let mut subtree_nodes = Vec::new();

    queue.push_back(subtree_root);
    visited.insert(subtree_root);

    while let Some(current) = queue.pop_front() {
        subtree_nodes.push(current);

        let children: Vec<NodeId> = cpg
            .get_outgoing_edges(current)
            .iter()
            .filter(|edge| edge.type_ == EdgeType::SyntaxChild)
            .map(|edge| edge.to)
            .collect();

        for child in children {
            if !visited.contains(&child) {
                visited.insert(child);
                queue.push_back(child);
            }
        }
    }

    debug!("Found {} nodes in subtree", subtree_nodes.len());

    for &node_id in &subtree_nodes {
        if let Some(node) = cpg.get_node_by_id(&node_id) {
            match node.type_ {
                NodeType::Function { .. } => {
                    compute_function_control_flow(cpg, node_id)?;
                }
                _ => {}
            }
        }
    }

    Ok(())
}

/// Compute control flow within a single function
fn compute_function_control_flow(cpg: &mut Cpg, function_node: NodeId) -> Result<(), String> {
    debug!("Computing control flow for function: {:?}", function_node);

    let statements = collect_statements_in_function(cpg, function_node);

    if statements.is_empty() {
        debug!("No statements found in function");
        return Ok(());
    }

    for i in 0..statements.len() {
        debug!("{}/{}", i + 1, statements.len());
        let current_stmt = statements[i];

        if let Some(node) = cpg.get_node_by_id(&current_stmt) {
            match &node.type_ {
                NodeType::Conditional {
                    condition,
                    then_branch,
                    else_branch,
                } => {
                    handle_if_statement_control_flow(
                        cpg,
                        current_stmt,
                        &statements,
                        i,
                        condition.clone(),
                        then_branch.clone(),
                        else_branch.clone(),
                    )?;
                }
                NodeType::Loop { condition, body } => {
                    handle_loop_control_flow(cpg, current_stmt, &statements, i)?;
                }
                NodeType::Return => {
                    debug!("Found return statement: {:?}", current_stmt);
                }
                NodeType::Block => {
                    handle_block_control_flow(cpg, current_stmt, &statements, i)?;
                }
                NodeType::Statement | NodeType::Expression | NodeType::Call => {
                    if i + 1 < statements.len() {
                        let next_stmt = statements[i + 1];
                        add_control_flow_edge(
                            cpg,
                            current_stmt,
                            next_stmt,
                            EdgeType::ControlFlowEpsilon,
                        )?;
                    }
                }
                _ => {
                    // For other node types, just connect sequentially
                    // Note: This may not be necessary or ideal considering we have no idea what the "Language Implementation" nodes are doing
                    if i + 1 < statements.len() {
                        let next_stmt = statements[i + 1];
                        add_control_flow_edge(
                            cpg,
                            current_stmt,
                            next_stmt,
                            EdgeType::ControlFlowEpsilon,
                        )?;
                    }
                }
            }
        }
    }

    Ok(())
}

fn collect_statements_in_function(cpg: &Cpg, function_node: NodeId) -> Vec<NodeId> {
    let mut statements = Vec::new();
    collect_statements_recursive(cpg, function_node, &mut statements);
    statements
}

fn collect_statements_recursive(cpg: &Cpg, node_id: NodeId, statements: &mut Vec<NodeId>) {
    if let Some(node) = cpg.get_node_by_id(&node_id) {
        if is_statement_node(&node.type_) {
            statements.push(node_id);
        }
    }

    let ordered_children = cpg.ordered_syntax_children(node_id);
    for child in ordered_children {
        collect_statements_recursive(cpg, child, statements);
    }
}

fn handle_if_statement_control_flow(
    cpg: &mut Cpg,
    if_stmt: NodeId,
    statements: &[NodeId],
    current_index: usize,
    condition_traversal: DescendantTraversal,
    then_branch_traversal: DescendantTraversal,
    else_branch_traversal: DescendantTraversal,
) -> Result<(), String> {
    debug!("Handling if statement control flow: {:?}", if_stmt);

    // Find the condition, then branch, and else branch (if any)
    let children = cpg.ordered_syntax_children(if_stmt);

    let condition = None; //find_child_by_type(cpg, &children, NodeType::Condition);
    let then_branch = None; //find_child_by_type(cpg, &children, NodeType::ThenBranch);
    let else_branch = None; // find_child_by_type(cpg, &children, NodeType::ElseBranch);

    if let (Some(condition), Some(then_branch)) = (condition, then_branch) {
        // TRUE
        add_control_flow_edge(cpg, condition, then_branch, EdgeType::ControlFlowTrue)?;

        // FALSE
        if let Some(else_branch) = else_branch {
            add_control_flow_edge(cpg, condition, else_branch, EdgeType::ControlFlowFalse)?;

            if current_index + 1 < statements.len() {
                let next_stmt = statements[current_index + 1];
                add_control_flow_edge(cpg, else_branch, next_stmt, EdgeType::ControlFlowEpsilon)?;
            }
        } else {
            if current_index + 1 < statements.len() {
                let next_stmt = statements[current_index + 1];
                add_control_flow_edge(cpg, condition, next_stmt, EdgeType::ControlFlowFalse)?;
            }
        }

        if current_index + 1 < statements.len() {
            let next_stmt = statements[current_index + 1];
            add_control_flow_edge(cpg, then_branch, next_stmt, EdgeType::ControlFlowEpsilon)?;
        }
    }

    Ok(())
}

fn handle_loop_control_flow(
    cpg: &mut Cpg,
    loop_stmt: NodeId,
    statements: &[NodeId],
    current_index: usize,
) -> Result<(), String> {
    debug!("Handling loop control flow: {:?}", loop_stmt);

    let children = cpg.ordered_syntax_children(loop_stmt);

    let condition = None; //find_child_by_type(cpg, &children, NodeType::Condition);
    let body = None; // find_child_by_type(cpg, &children, NodeType::LoopBody);
    let init = None; // find_child_by_type(cpg, &children, NodeType::LoopInit);
    let update = None; // find_child_by_type(cpg, &children, NodeType::LoopUpdate);

    if let (Some(condition), Some(body)) = (condition, body) {
        // For loops: init -> condition
        if let Some(init) = init {
            add_control_flow_edge(cpg, init, condition, EdgeType::ControlFlowEpsilon)?;
        }

        // True edge from condition to body
        add_control_flow_edge(cpg, condition, body, EdgeType::ControlFlowTrue)?;

        // Body flows to update (for loops) or back to condition (while loops)
        if let Some(update) = update {
            // For loop: body -> update -> condition
            add_control_flow_edge(cpg, body, update, EdgeType::ControlFlowEpsilon)?;
            add_control_flow_edge(cpg, update, condition, EdgeType::ControlFlowEpsilon)?;
        } else {
            // While loop: body -> condition
            add_control_flow_edge(cpg, body, condition, EdgeType::ControlFlowEpsilon)?;
        }

        // False edge from condition to next statement (loop exit)
        if current_index + 1 < statements.len() {
            let next_stmt = statements[current_index + 1];
            add_control_flow_edge(cpg, condition, next_stmt, EdgeType::ControlFlowFalse)?;
        }
    }

    Ok(())
}

fn handle_block_control_flow(
    cpg: &mut Cpg,
    block_stmt: NodeId,
    statements: &[NodeId],
    current_index: usize,
) -> Result<(), String> {
    debug!("Handling block control flow: {:?}", block_stmt);

    // First statement in block
    if let Some(first_child) = get_first_statement_in_block(cpg, block_stmt) {
        add_control_flow_edge(cpg, block_stmt, first_child, EdgeType::ControlFlowEpsilon)?;
    }

    // Connect last statement in block to next statement after the block
    if current_index + 1 < statements.len() {
        let next_stmt = statements[current_index + 1];
        if let Some(last_child) = get_last_statement_in_block(cpg, block_stmt) {
            add_control_flow_edge(cpg, last_child, next_stmt, EdgeType::ControlFlowEpsilon)?;
        }
    }

    Ok(())
}

/// Idempotent computation of the data dependence for a subtree in the CPG.
/// This pass assumes that the control flow has already been computed.
pub fn data_dep_pass(_cpg: &mut Cpg, _subtree_root: NodeId) -> Result<(), String> {
    // TODO: Implement data dependence analysis

    Ok(())
}

// --- Tests --- //

#[cfg(test)]
mod tests {
    use super::*;

    fn check_generic_features(name: &str) {
        let lang: RegisteredLanguage = name.parse().expect("Failed to parse language");
        assert!(
            lang.get_variant_names()
                .contains(&name.to_lowercase().as_str()),
            "Language {} not found in variants",
            name
        );
        assert_eq!(
            lang.get_display_name(),
            name,
            "Display name mismatch for {}",
            name
        );
        assert!(
            lang.get_parser().is_ok(),
            "Failed to get parser for {}",
            name
        );
    }

    #[test]
    fn test_c_features() {
        check_generic_features("C");
    }
}
