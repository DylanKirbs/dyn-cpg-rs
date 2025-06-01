use std::collections::HashMap;

use tracing::{debug, warn};
use tracing_subscriber::field::debug;

use crate::cpg::{Edge, EdgeType, Node, NodeId, NodeType};

use super::super::define_language;
use super::{Cpg, Language};

define_language! {
    C, ["c", "h", "c++", "cpp", "clang"], tree_sitter_c::LANGUAGE, cst_to_cpg
}

pub fn cst_to_cpg(lang: &C, tree: tree_sitter::Tree) -> Result<Cpg, String> {
    debug!("Converting CST to CPG for {:?}", lang);

    let mut cpg = Cpg::new(None, None);
    let mut cursor = tree.walk();

    translate(&mut cpg, &mut cursor).map_err(|e| {
        warn!("Failed to translate CST: {}", e);
        e
    })?;

    Ok(cpg)
}

fn map_node_kind(ts_node: &tree_sitter::Node) -> NodeType {
    match ts_node.kind() {
        "translation_unit" => NodeType::TranslationUnit,

        "function_definition" | "function_declarator" => NodeType::Function,

        "identifier" => NodeType::Identifier,

        "if_statement" | "for_statement" | "expression_statement" => NodeType::Statement,

        "call_expression" => NodeType::Call,

        "compound_statement" => NodeType::Block,

        "return_statement" => NodeType::Return,

        "primitive_type" | "type_identifier" => NodeType::Type,

        "field_identifier" | "field_declarator" => NodeType::Identifier,

        "subscript_expression"
        | "parenthesized_expression"
        | "parameter_list"
        | "argument_list"
        | "number_literal"
        | "string_literal"
        | "field_expression"
        | "binary_expression"
        | "string_content"
        | "escape_sequence"
        | "init_declarator"
        | "conditional_expression" => NodeType::Expression,

        "comment" => NodeType::Comment,

        // Since TS returns a CST, we have the whole parse tree, so named tokens are lumped into language implementation nodes
        other => NodeType::LanguageImplementation(other.to_string()),
    }
}

fn translate(cpg: &mut Cpg, cursor: &mut tree_sitter::TreeCursor) -> Result<NodeId, String> {
    let node = cursor.node();

    let id = format!("node_{}", node.id());
    let type_ = map_node_kind(&node);

    debug!("Type: {:?}", type_);

    cpg.add_node(Node {
        id: id.clone(),
        type_: type_,
        properties: HashMap::new(),
    });

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
