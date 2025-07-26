use super::super::define_language;
use super::Language;
use crate::cpg::NodeType;

define_language! {
    C, ["c", "h", "c++", "cpp", "clang"], tree_sitter_c::LANGUAGE, map_node_kind
}

fn map_node_kind(_: &C, node_kind: &'static str) -> NodeType {
    match node_kind {
        "translation_unit" => NodeType::TranslationUnit,

        "function_declarator" => NodeType::Function,

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
