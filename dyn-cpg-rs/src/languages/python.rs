use tracing::debug;

use super::super::define_language;
use super::{Cpg, Language};

define_language! {
    Python, ["python", "py", "python3"], tree_sitter_python::LANGUAGE, cst_to_cpg
}

pub fn cst_to_cpg(lang: &Python, tree: tree_sitter::Tree) -> Result<Cpg, String> {
    debug!("{:?} {:?}", lang, tree);
    Err(format!("Unimplemented for {:?}", lang))
}
