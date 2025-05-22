use std::str::FromStr;
use strum::VariantNames;
use strum_macros::EnumVariantNames;
use tree_sitter::Parser as TSParser;

// --- Language & Parser --- //

#[derive(Debug, Clone, EnumVariantNames)]
pub enum Language {
    Java,
    Python,
    C,
}

impl FromStr for Language {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "java" => Ok(Language::Java),
            "python" | "py" | "python3" => Ok(Language::Python),
            "c" | "h" => Ok(Language::C),
            _ => Err(format!(
                "Unknown language '{}', expected one of {:?}",
                s,
                Language::VARIANTS
            )),
        }
    }
}

impl Language {
    pub fn get_parser(&self) -> Result<TSParser, String> {
        let mut parser = TSParser::new();

        let res = match self {
            Language::Java => parser.set_language(&tree_sitter_java::LANGUAGE.into()),
            Language::Python => parser.set_language(&tree_sitter_python::LANGUAGE.into()),
            Language::C => parser.set_language(&tree_sitter_c::LANGUAGE.into()),
        };

        match res {
            Ok(_) => Ok(parser),
            Err(e) => Err(format!("Failed to set language parser: {}", e)),
        }
    }
}
