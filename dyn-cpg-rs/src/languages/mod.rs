/// This module defines the `Language` trait and provides macros to register languages.
/// Each language should be defined in its own module and implement the `Language` trait, a macro has been provided to simplify this process.
use crate::cpg::{Cpg, Edge, EdgeType, IdenType, Node, NodeId, NodeType};
use tracing::{debug, trace, warn};
mod c;
use c::C;

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
macro_rules! define_language {(
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
    debug!("[CPG TO CST] Converting CST to CPG for {:?}", lang);

    let mut cpg = Cpg::new(lang.clone(), source);
    let mut cursor = tree.walk();

    translate(&mut cpg, &mut cursor).map_err(|e| {
        warn!("[CPG TO CST] Failed to translate CST: {}", e);
        e
    })?;

    let root_opt = cpg.get_root();

    if root_opt.is_none() {
        return Err("CST translation resulted in an empty CPG".to_string());
    }

    let root = root_opt.expect("Root node must be present if it is not None");

    cf_pass(&mut cpg, root).map_err(|e| {
        warn!("[CPG TO CST] Failed to compute control flow: {}", e);
        e
    })?;

    data_dep_pass(&mut cpg, root).map_err(|e| {
        warn!("[CPG TO CST] Failed to compute data dependence: {}", e);
        e
    })?;

    Ok(cpg)
}

pub fn translate(cpg: &mut Cpg, cursor: &mut tree_sitter::TreeCursor) -> Result<NodeId, String> {
    let cst_node = cursor.node();
    let (cpg_node_id, cpg_node_type) = pre_translate_node(cpg, &cst_node)?;

    if cursor.goto_first_child() {
        let mut left_child_id: Option<NodeId> = None;

        loop {
            let child_id = translate(cpg, cursor)?;

            // Edge from parent to child
            cpg.add_edge(Edge {
                from: cpg_node_id,
                to: child_id,
                type_: EdgeType::SyntaxChild,
            });

            // Edge from left sibling to current
            if let Some(left_id) = &left_child_id {
                cpg.add_edge(Edge {
                    from: *left_id,
                    to: child_id,
                    type_: EdgeType::SyntaxSibling,
                });
            }

            left_child_id = Some(child_id);

            if !cursor.goto_next_sibling() {
                break;
            }
        }

        cursor.goto_parent();
    }

    post_translate_node(cpg, cpg_node_type, cpg_node_id, &cst_node);

    Ok(cpg_node_id)
}

pub fn pre_translate_node(
    cpg: &mut Cpg,
    cst_node: &tree_sitter::Node,
) -> Result<(NodeId, NodeType), String> {
    let source_len = cpg.get_source().len();

    // Validate node range is within source bounds
    if cst_node.start_byte() > source_len || cst_node.end_byte() > source_len {
        return Err(format!(
            "Node range ({}, {}) exceeds source length {}",
            cst_node.start_byte(),
            cst_node.end_byte(),
            source_len
        ));
    }

    let type_ = cpg.get_language().clone().map_node_kind(cst_node.kind());

    let cpg_node = Node {
        type_: type_.clone(),
        raw_type: cst_node.kind().to_string(),
        ..Default::default()
    };

    let cpg_node_id = cpg.add_node(cpg_node, cst_node.start_byte(), cst_node.end_byte());

    trace!(
        "[PRE TRANSLATE NODE] Created {:?} kind={} range=({}, {})",
        cpg_node_id,
        cst_node.kind(),
        cst_node.start_byte(),
        cst_node.end_byte()
    );

    Ok((cpg_node_id, type_))
}

pub fn post_translate_node(
    cpg: &mut Cpg,
    type_: NodeType,
    cpg_node_id: NodeId,
    cst_node: &tree_sitter::Node,
) {
    trace!(
        "[POST TRANSLATE NODE] Processing kind={} range=({}, {}) -> {:?}",
        cst_node.kind(),
        cst_node.start_byte(),
        cst_node.end_byte(),
        cpg_node_id
    );
    match type_ {
        // Functions get their names from their name_traversal & a special return node
        NodeType::Function { name_traversals } => {
            let mut found_name = false;
            for name_traversal in name_traversals {
                let id_node = name_traversal.get_descendent(cpg, &cpg_node_id);
                if let Some(id_node) = id_node {
                    let name = cpg.get_node_source(&id_node);

                    if name == "*" {
                        continue;
                    }

                    if let Some(n) = cpg.get_node_by_id_mut(&cpg_node_id) {
                        n.name = Some(name.clone());
                    }

                    trace!(
                        "[POST TRANSLATE NODE] Function node name found: {:?} {:?}",
                        cst_node.kind(),
                        name
                    );

                    found_name = true;
                    break;
                }
            }
            if !found_name {
                warn!(
                    "[POST TRANSLATE NODE] Function name traversal failed for node {:?}, attempting fallback",
                    cpg_node_id
                );

                // This is quite naive and just finds the first Indentifier descentant of the function
                let descendants = cpg.post_dfs_ordered_syntax_descendants(cpg_node_id);
                for child in descendants {
                    if let Some(child_node) = cpg.get_node_by_id(&child) {
                        if matches!(child_node.type_, NodeType::Identifier { .. }) {
                            let name = cpg.get_node_source(&child);
                            if let Some(n) = cpg.get_node_by_id_mut(&cpg_node_id) {
                                n.name = Some(name.clone());
                            }

                            trace!(
                                "[POST TRANSLATE NODE] Function node name found via fallback: {:?} {:?}",
                                cst_node.kind(),
                                name
                            );
                            break;
                        }
                    }
                }
            }

            // Add a CF Function Return to the function (only if one doesn't already exist)
            let existing_function_return = cpg
                .get_outgoing_edges(cpg_node_id)
                .iter()
                .find(|e| e.type_ == EdgeType::ControlFlowFunctionReturn)
                .map(|e| e.to);

            if existing_function_return.is_none() {
                if let Some((s, e)) = cpg.get_node_offsets_by_id(&cpg_node_id) {
                    let fn_end = cpg.add_node(
                        Node {
                            type_: NodeType::FunctionReturn,
                            ..Default::default()
                        },
                        s,
                        e,
                    );

                    cpg.add_edge(Edge {
                        from: cpg_node_id,
                        to: fn_end,
                        type_: EdgeType::ControlFlowFunctionReturn,
                    });
                }
            }
        }

        // Statements and Expressions get their reads and writes tracked
        NodeType::Statement | NodeType::Expression | NodeType::Return => {
            trace!(
                "[POST TRANSLATE NODE] Tracking reads and writes for: {:?} {:?}",
                cst_node.kind(),
                cpg_node_id
            );
            let mut assigned = vec![];
            let mut read = vec![];

            let descendants = cpg.post_dfs_ordered_syntax_descendants(cpg_node_id);
            for descendant in descendants {
                if let Some(child_node) = cpg.get_node_by_id(&descendant) {
                    if let NodeType::Identifier { type_: iden_type } = child_node.type_.clone() {
                        match iden_type {
                            IdenType::WRITE => {
                                assigned.push(child_node.name.clone().unwrap_or_default());
                            }
                            IdenType::READ => {
                                read.push(child_node.name.clone().unwrap_or_default());
                            }
                            IdenType::UNKNOWN => {
                                assigned.push(child_node.name.clone().unwrap_or_default());
                                read.push(child_node.name.clone().unwrap_or_default());
                            }
                        }
                    }
                }
            }

            // Clean up
            assigned.dedup();
            assigned.retain(|s| !s.is_empty());
            read.dedup();
            read.retain(|s| !s.is_empty());

            if let Some(n) = cpg.get_node_by_id_mut(&cpg_node_id) {
                if !assigned.is_empty() {
                    n.df_writes = assigned;
                }
                if !read.is_empty() {
                    n.df_reads = read;
                }
            }
        }

        // Identifiers get their name
        NodeType::Identifier { .. } => {
            if cpg
                .get_node_by_id(&cpg_node_id)
                .and_then(|n| n.name.clone())
                .is_none()
            {
                let iden_name = cpg.get_node_source(&cpg_node_id).clone();

                trace!(
                    "[POST TRANSLATE NODE] Identifier node name found: {:?} {:?}",
                    cst_node.kind(),
                    iden_name
                );

                if let Some(n) = cpg.get_node_by_id_mut(&cpg_node_id) {
                    n.name = Some(iden_name);
                }
            } else {
                // Force update the name during incremental updates
                let iden_name = cpg.get_node_source(&cpg_node_id).clone();
                let current_name = cpg
                    .get_node_by_id(&cpg_node_id)
                    .and_then(|n| n.name.clone())
                    .unwrap_or_default();

                trace!(
                    "[POST TRANSLATE NODE] Identifier node name update: {:?} current={:?} new={:?}",
                    cst_node.kind(),
                    current_name,
                    iden_name
                );

                if current_name != iden_name {
                    if let Some(n) = cpg.get_node_by_id_mut(&cpg_node_id) {
                        n.name = Some(iden_name);
                    }
                }
            }
        }

        _ => {}
    }
}

// -- Helpers -- //

fn should_descend(node_type: &NodeType) -> bool {
    matches!(
        node_type,
        NodeType::Statement
            | NodeType::Expression
            | NodeType::Call
            | NodeType::Block
            | NodeType::Branch { .. }
            | NodeType::Loop { .. }
            | NodeType::Function { .. }
            | NodeType::Return
    )
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
            trace!(
                "[CONTROL FLOW] Edge {{{:?} -> [{:?}] -> {:?}}} exists",
                from, edge_type, to
            );
            return Ok(());
        }
    }

    trace!(
        "[CONTROL FLOW] Edge {{{:?} -> [{:?}] -> {:?}}} created",
        from, edge_type, to
    );
    cpg.add_edge(Edge {
        from,
        to,
        type_: edge_type,
    });

    Ok(())
}

fn add_data_dep_edge(
    cpg: &mut Cpg,
    from: NodeId,
    to: NodeId,
    edge_type: EdgeType,
) -> Result<(), String> {
    // Don't create self-loops for data dependence
    if from == to {
        trace!(
            "[DATA DEPENDENCE] Edge {{{:?} -> [{:?}] -> {:?}}} skipped (self-loop)",
            from, edge_type, to
        );
        return Ok(());
    }

    let existing_edges = cpg.get_outgoing_edges(from);
    for edge in existing_edges {
        if edge.to == to && edge.type_ == edge_type {
            trace!(
                "[DATA DEPENDENCE] Edge {{{:?} -> [{:?}] -> {:?}}} skipped (exists)",
                from, edge_type, to
            );

            return Ok(());
        }
    }

    trace!(
        "[DATA DEPENDENCE] Edge {{{:?} -> [{:?}] -> {:?}}} created",
        from, edge_type, to
    );
    cpg.add_edge(Edge {
        from,
        to,
        type_: edge_type,
    });

    Ok(())
}

fn get_descent_children(cpg: &Cpg, node_id: NodeId) -> Vec<NodeId> {
    let children = cpg.ordered_syntax_children(node_id);
    children
        .into_iter()
        .filter(|&child_id| {
            if let Some(child) = cpg.get_node_by_id(&child_id) {
                should_descend(&child.type_)
            } else {
                false
            }
        })
        .collect()
}

fn find_first_processable_descendant(cpg: &Cpg, node_id: NodeId) -> Option<NodeId> {
    if let Some(node) = cpg.get_node_by_id(&node_id) {
        if should_descend(&node.type_) {
            return Some(node_id);
        }
    }

    let children = cpg.ordered_syntax_children(node_id);
    for child in children {
        if let Some(descendant) = find_first_processable_descendant(cpg, child) {
            return Some(descendant);
        }
    }

    None
}

/// Navigate up the CPG via SyntaxChild edges until we get to a parent Block, Function or TranslationUnit
/// If the node is already one of these, return it directly.
pub fn get_container_parent(cpg: &Cpg, node_id: NodeId) -> NodeId {
    match cpg.get_node_by_id(&node_id) {
        Some(node)
            if matches!(
                node.type_,
                NodeType::Block | NodeType::Function { .. } | NodeType::TranslationUnit
            ) =>
        {
            return node_id; // Already a container
        }
        _ => {}
    }

    let par = cpg
        .get_incoming_edges(node_id)
        .into_iter()
        .find(|e| e.type_ == EdgeType::SyntaxChild)
        .map(|e| e.from);

    match par {
        Some(id) => {
            let node = cpg.get_node_by_id(&id).expect("Node must exist");
            if matches!(
                node.type_,
                NodeType::Block | NodeType::Function { .. } | NodeType::TranslationUnit
            ) {
                id
            } else {
                get_container_parent(cpg, id)
            }
        }
        None => {
            warn!(
                "[CONTROL FLOW] No container parent found for node: {:?}",
                node_id
            );
            node_id // Fallback to the original node if no parent found
        }
    }
}

pub fn get_containing_function(cpg: &Cpg, node_id: NodeId) -> Option<NodeId> {
    let mut current = node_id;
    while let Some(node) = cpg.get_node_by_id(&current) {
        if matches!(node.type_, NodeType::Function { .. }) {
            return Some(current);
        }
        current = cpg
            .get_incoming_edges(current)
            .into_iter()
            .find(|e| e.type_ == EdgeType::SyntaxChild)
            .map(|e| e.from)?;
    }
    None
}

// --- Control Flow & Data Dependence --- //

/// Idempotent computation of the control flow for a subtree in the CPG.
/// This pass assumes that the AST has been construed into a CPG.
pub fn cf_pass(cpg: &mut Cpg, subtree_root: NodeId) -> Result<(), String> {
    debug!(
        "[CONTROL FLOW] Starting control flow pass on subtree rooted at {:?}",
        subtree_root
    );
    compute_control_flow_postorder(
        cpg,
        get_container_parent(cpg, subtree_root),
        get_containing_function(cpg, subtree_root),
    )?;
    debug!(
        "[CONTROL FLOW] Control flow pass completed on subtree rooted at {:?}",
        subtree_root
    );
    Ok(())
}

/// Control flow computation
/// Returns the exit points (last reachable statements) of this subtree
fn compute_control_flow_postorder(
    cpg: &mut Cpg,
    node_id: NodeId,
    parent_function: Option<NodeId>,
) -> Result<Vec<NodeId>, String> {
    let node = cpg
        .get_node_by_id(&node_id)
        .ok_or_else(|| format!("Node not found: {:?}", node_id))?;

    match &node.type_ {
        NodeType::Branch {
            condition,
            then_branch,
            else_branch,
        } => {
            let condition = condition.clone().get_descendent(cpg, &node_id);
            let then_block = then_branch.clone().get_descendent(cpg, &node_id);
            let else_block = else_branch.clone().get_descendent(cpg, &node_id);

            let condition = condition.ok_or_else(|| {
                format!(
                    "Branch missing condition: {:?} [{:?}]",
                    node_id,
                    cpg.get_node_source(&node_id),
                )
            })?;
            let then_block = then_block.ok_or_else(|| {
                format!(
                    "Branch missing then block: {:?} [{:?}]",
                    node_id,
                    cpg.get_node_source(&node_id),
                )
            })?;

            trace!(
                "[CONTROL FLOW] [BRANCH] condition: {:?}, then: {:?}, else: {:?}",
                condition, then_block, else_block
            );

            // Process condition first
            let _condition_exits = compute_control_flow_postorder(cpg, condition, parent_function)?;

            add_control_flow_edge(cpg, node_id, condition, EdgeType::ControlFlowEpsilon)?;

            // Add control flow edges from condition to branches
            add_control_flow_edge(cpg, condition, then_block, EdgeType::ControlFlowTrue)?;

            let mut branch_exits = Vec::new();

            // Process then branch
            let then_exits = compute_control_flow_postorder(cpg, then_block, parent_function)?;
            branch_exits.extend(then_exits);

            // Process else branch if present
            if let Some(else_block) = else_block {
                add_control_flow_edge(cpg, condition, else_block, EdgeType::ControlFlowFalse)?;
                let else_exits = compute_control_flow_postorder(cpg, else_block, parent_function)?;
                branch_exits.extend(else_exits);
            } else {
                // No else branch - condition can fall through
                branch_exits.push(condition);
            }

            Ok(branch_exits)
        }

        NodeType::Loop { condition, body } => {
            // TODO: Handle break/continue statements properly

            let condition = condition.clone().get_descendent(cpg, &node_id);
            let body = body.clone().get_descendent(cpg, &node_id);

            let condition = condition.ok_or_else(|| {
                format!(
                    "Loop missing condition: {:?} [{:?}]",
                    node_id,
                    cpg.get_node_source(&node_id),
                )
            })?;
            let body = body.ok_or_else(|| {
                format!(
                    "Loop missing body: {:?} [{:?}]",
                    node_id,
                    cpg.get_node_source(&node_id),
                )
            })?;

            trace!(
                "[CONTROL FLOW] [LOOP] condition: {:?}, body: {:?}",
                condition, body
            );

            // Process condition
            let _condition_exits = compute_control_flow_postorder(cpg, condition, parent_function)?;

            // Process body
            let body_exits = compute_control_flow_postorder(cpg, body, parent_function)?;

            // Add control flow edges
            add_control_flow_edge(cpg, condition, body, EdgeType::ControlFlowTrue)?;

            // Connect body exits back to condition
            for &body_exit in &body_exits {
                add_control_flow_edge(cpg, body_exit, condition, EdgeType::ControlFlowEpsilon)?;
            }

            // Loop exit is the condition (false branch)
            Ok(vec![condition])
        }

        NodeType::Return => {
            if let Some(par) = parent_function {
                let outgoing_edges: Vec<_> = cpg.get_outgoing_edges(par).into_iter().collect();
                let targets: Vec<_> = outgoing_edges
                    .iter()
                    .filter(|e| e.type_ == EdgeType::ControlFlowFunctionReturn)
                    .map(|e| e.to)
                    .collect();
                for target in targets {
                    add_control_flow_edge(cpg, node_id, target, EdgeType::ControlFlowEpsilon)?;
                }
            }
            // Return statements don't have exit points (they terminate) [OR DO THEY? VSauce]
            Ok(vec![])
        }

        NodeType::Function { .. } => {
            let statement_children = get_descent_children(cpg, node_id);
            let mut exits = Vec::new();

            if !statement_children.is_empty() {
                // Find where execution actually starts (leftmost descendant of first child)
                let execution_start = find_execution_entry_point(cpg, statement_children[0]);
                add_control_flow_edge(cpg, node_id, execution_start, EdgeType::ControlFlowEpsilon)?;
                exits = process_sequential_statements(cpg, &statement_children, Some(node_id))?;
            } else {
                exits.push(node_id);
            }

            // patch exits to function return node
            let outgoing_edges = cpg.get_outgoing_edges(node_id);
            let fn_return = outgoing_edges
                .iter()
                .find(|e| e.type_ == EdgeType::ControlFlowFunctionReturn)
                .map(|e| e.to);

            if let Some(fn_ret) = fn_return {
                for exit in &exits {
                    add_control_flow_edge(cpg, *exit, fn_ret, EdgeType::ControlFlowEpsilon)?;
                }
            }

            Ok(vec![])
        }

        // For other statement types (expressions, calls, etc.)
        _ if should_descend(&node.type_) => {
            let statement_children = get_descent_children(cpg, node_id);
            if !statement_children.is_empty() {
                let exits =
                    process_sequential_statements(cpg, &statement_children, parent_function)?;

                for &exit in &exits {
                    add_control_flow_edge(cpg, exit, node_id, EdgeType::ControlFlowEpsilon)?;
                }

                Ok(vec![node_id])
            } else {
                Ok(vec![node_id])
            }
        }

        _ => {
            let non_seq_children = cpg.ordered_syntax_children(node_id);
            for node in &non_seq_children {
                if let Some(stmt_child) = find_first_processable_descendant(cpg, *node) {
                    compute_control_flow_postorder(cpg, stmt_child, parent_function)?;
                }
            }
            Ok(vec![])
        }
    }
}

/// Process a sequence of statements, connecting them with epsilon edges
/// Returns the exit points of the last reachable statement
fn process_sequential_statements(
    cpg: &mut Cpg,
    statements: &[NodeId],
    parent_function: Option<NodeId>,
) -> Result<Vec<NodeId>, String> {
    if statements.is_empty() {
        return Ok(vec![]);
    }

    let mut prev_exits = compute_control_flow_postorder(cpg, statements[0], parent_function)?;

    for window in statements.windows(2) {
        let next_sibling = window[1];

        // Find where execution actually starts in the next sibling
        let next_entry = find_execution_entry_point(cpg, next_sibling);

        for &exit in &prev_exits {
            add_control_flow_edge(cpg, exit, next_entry, EdgeType::ControlFlowEpsilon)?;
        }
        prev_exits = compute_control_flow_postorder(cpg, next_sibling, parent_function)?;
    }

    Ok(prev_exits)
}

fn find_execution_entry_point(cpg: &Cpg, node_id: NodeId) -> NodeId {
    let children = get_descent_children(cpg, node_id);
    if !children.is_empty() {
        // Execution starts at the first child
        find_execution_entry_point(cpg, children[0])
    } else {
        // No processable children, execution starts at this node
        node_id
    }
}

/// Idempotent computation of the data dependence for a subtree in the CPG.
/// This pass assumes that the control flow has already been computed.
pub fn data_dep_pass(cpg: &mut Cpg, subtree_root: NodeId) -> Result<(), String> {
    debug!("[DATA DEPENDENCE] Starting data dependence pass");

    let all_nodes = cpg.post_dfs_ordered_syntax_descendants(subtree_root);

    // Map from variable name to last write node id
    use std::collections::HashMap;
    let mut last_write: HashMap<String, NodeId> = HashMap::new();

    // TODO: Make this actually consider control flow (i.e. when branches exists, dependence should not go from the one branch into the other)
    // TODO: CFFunctionReturn is DDep on all nodes with CFEdges that flow into it
    // For each node in control flow order, track writes and add edges to reads
    // This is a naive approach: just walk all nodes in DFS order
    for &curr_node_id in &all_nodes {
        let node = match cpg.get_node_by_id(&curr_node_id) {
            Some(n) => n.clone(),
            None => continue,
        };
        // Check for variable read
        if !node.df_reads.is_empty() {
            trace!(
                "[DATA DEPENDENCE] Node {:?} reads vars: {}",
                curr_node_id,
                node.df_reads.join(", ")
            );
        }
        for var in node.df_reads.iter() {
            if let Some(&last_written_id) = last_write.get(var) {
                add_data_dep_edge(
                    cpg,
                    last_written_id,
                    curr_node_id,
                    EdgeType::PDData(var.to_string()),
                )?;
            }
        }

        // Check for variable write (assignment)
        if !node.df_writes.is_empty() {
            trace!(
                "[DATA DEPENDENCE] Node {:?} assigns vars: {}",
                curr_node_id,
                node.df_writes.join(", ")
            );
        }
        for v in node.df_writes.iter() {
            last_write.insert(v.to_string(), curr_node_id);
        }
    }
    debug!("[DATA DEPENDENCE] Data dependence pass completed");

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
