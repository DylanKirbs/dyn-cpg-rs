use std::collections::HashMap;

use tracing::debug;

use crate::cpg::{Node, NodeType};

use super::super::define_language;
use super::{Cpg, Language};

define_language! {
    C, ["c", "h", "c++", "cpp", "clang"], tree_sitter_c::LANGUAGE, cst_to_cpg
}

pub fn cst_to_cpg(lang: &C, tree: tree_sitter::Tree) -> Result<Cpg, String> {
    debug!("Converting CST to CPG for {:?}", lang);

    let mut cpg = Cpg::new(None, None);

    cpg.add_node(Node {
        id: "root".to_string(),
        type_: NodeType::TranslationUnit,
        properties: HashMap::new(),
    });

    Err("Conversion not completed".to_string())
}

fn translate(cpg: &mut Cpg, node: tree_sitter::Node) -> Result<(), String> {
    debug!("Translating node: {:?}", node);

    let id = format!("node_{}", node.id());
    let type_ = NodeType::Unknown;

    cpg.add_node(Node {
        id,
        type_,
        properties: HashMap::new(),
    });

    Ok(())
}
