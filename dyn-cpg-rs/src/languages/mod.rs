use std::collections::HashMap;

/// This module defines the `Language` trait and provides macros to register languages.
/// Each language should be defined in its own module and implement the `Language` trait, a macro has been provided to simplify this process.
use crate::cpg::{Cpg, Edge, EdgeType, Node, NodeId, NodeType};

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
        $name:ident, [$($variant_names:expr),+], $lang:path, $kind:expr
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
                    $kind(self, node_kind)
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

            pub fn cst_to_cpg(&self, tree: tree_sitter::Tree) -> Result<Cpg, String> {
                match self {
                    $(RegisteredLanguage::$variant(_) => cst_to_cpg(&self, tree),)+
                }
            }
        }
    };
}

// --- Language Definitions --- //

register_languages! { C }

// --- Generic Language Utilities --- //

pub fn cst_to_cpg(lang: &RegisteredLanguage, tree: tree_sitter::Tree) -> Result<Cpg, String> {
    debug!("Converting CST to CPG for {:?}", lang);

    let mut cpg = Cpg::new(lang.clone());
    let mut cursor = tree.walk();

    translate(&mut cpg, &mut cursor).map_err(|e| {
        warn!("Failed to translate CST: {}", e);
        e
    })?;

    Ok(cpg)
}

fn translate(cpg: &mut Cpg, cursor: &mut tree_sitter::TreeCursor) -> Result<NodeId, String> {
    let node = cursor.node();

    let type_ = cpg.get_language().map_node_kind(node.kind());

    let id = cpg.add_node(
        Node {
            type_,
            properties: HashMap::new(),
        },
        node.start_byte(),
        node.end_byte(),
    );

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
