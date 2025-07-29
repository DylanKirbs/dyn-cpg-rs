use super::super::define_language;
use super::Language;
use crate::{
    cpg::{DescendantTraversal, NodeType},
    desc_trav,
};

define_language! {
    C, ["c", "h", "c++", "cpp", "clang"], tree_sitter_c::LANGUAGE, map_node_kind
}

fn map_node_kind(_: &C, node_kind: &'static str) -> NodeType {
    match node_kind {
        "translation_unit" => NodeType::TranslationUnit,

        "function_definition" => NodeType::Function {
            name_traversal: desc_trav![1, 0],
            name: None,
        },

        "identifier" => NodeType::Identifier,

        "if_statement" => NodeType::Branch {
            condition: desc_trav![1],
            then_branch: desc_trav![2],
            else_branch: desc_trav![3],
        }, // TODO: Figure out switch

        "for_statement" => NodeType::Loop {
            condition: desc_trav![3],
            body: desc_trav![7],
        },
        "while_statement" => NodeType::Loop {
            condition: desc_trav![1],
            body: desc_trav![2],
        },

        "break" | "continue" => NodeType::Statement, // TODO: Possible add more specific node type to handle this

        "expression_statement" | "declaration" => NodeType::Statement,

        "call_expression" => NodeType::Call,

        "compound_statement" => NodeType::Block,

        "return_statement" => NodeType::Return,

        "primitive_type" | "type_identifier" => NodeType::Type,

        "field_identifier" | "field_declarator" => NodeType::Identifier,

        "subscript_expression"
        | "parenthesized_expression"
        | "parameter_list"
        | "argument_list"
        | "field_expression"
        | "assignment_expression"
        | "binary_expression"
        | "update_expression"
        | "init_declarator"
        | "conditional_expression" => NodeType::Expression,

        "comment" => NodeType::Comment,

        // Since TS returns a CST, we have the whole parse tree, so named tokens are lumped into language implementation nodes
        other => NodeType::LanguageImplementation(other.to_string()),
    }
}
