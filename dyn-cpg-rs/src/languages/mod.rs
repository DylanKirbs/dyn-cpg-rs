/// This module defines the `Language` trait and provides macros to register languages.
/// Each language should be defined in its own module and implement the `Language` trait, a macro has been provided to simplify this process.
use crate::cpg::Cpg;

mod c;
mod python;

use c::C;
use python::Python;

// --- The Language Trait and Construction Macros --- //

trait Language: Default + std::fmt::Debug {
    /// The display name of the language.
    const DISPLAY_NAME: &'static str;

    /// The variant names and extensions for the language.
    const VARIANT_NAMES: &'static [&'static str];

    /// Get a Tree-sitter parser for the language.
    /// Each call to this function returns a new parser instance.
    fn get_parser(&self) -> Result<tree_sitter::Parser, String>;

    /// Convert a Tree-sitter syntax tree (CST) to a Code Property Graph (CPG).
    fn cst_to_cpg(&self, tree: tree_sitter::Tree) -> Result<Cpg, String>;
}

#[macro_export]
/// Macro to define a new language complying with the `Language` trait.
macro_rules! define_language {
    (
        $name:ident, [$($variant_names:expr),+], $lang:path, $cpg:expr
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

                fn cst_to_cpg(&self, tree: tree_sitter::Tree) -> Result<Cpg, String> {
                    $cpg(&self, tree)
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

            pub fn cst_to_cpg(&self, tree: tree_sitter::Tree) -> Result<Cpg, String> {
                match self {
                    $(RegisteredLanguage::$variant(language) => language.cst_to_cpg(tree),)+
                }
            }
        }
    };
}

// --- Language Definitions --- //

register_languages! { Python, C }

// --- Tests --- //

#[cfg(test)]
mod tests {
    use super::*;

    fn check_generic_features(name: &str) {
        let lang: RegisteredLanguage = name.parse().expect("Failed to parse language");
        assert_eq!(
            lang.get_variant_names()
                .contains(&name.to_lowercase().as_str()),
            true,
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
    fn test_python_features() {
        check_generic_features("Python");
    }

    #[test]
    fn test_c_features() {
        check_generic_features("C");
    }
}
