use tree_sitter::Parser as TSParser;

// --- The Language Trait and Construction Macro --- //

pub trait Language: std::fmt::Debug + Clone {
    /// Get the display name of the language.
    fn get_display_name(&self) -> &'static str;

    /// Get the variant names and extensions for the language.
    fn get_variant_names(&self) -> &'static [&'static str];

    /// Get the Tree-sitter parser for the language.
    fn get_parser(&self) -> Result<TSParser, String>;
}

macro_rules! define_languages {
    (
        $(
            $variant:ident => $type:ident, [$($alias:expr),+], $lang:path;
        )+
    ) => {
        $(
            #[derive(Debug, Clone)]
            pub struct $type;

            impl Default for $type {
                fn default() -> Self {
                    Self
                }
            }

            impl Language for $type {
                fn get_display_name(&self) -> &'static str {
                    stringify!($variant)
                }

                fn get_variant_names(&self) -> &'static [&'static str] {
                    &[$($alias),+]
                }

                fn get_parser(&self) -> Result<tree_sitter::Parser, String> {
                    let mut parser = tree_sitter::Parser::new();
                    parser.set_language(&($lang).into())
                        .map(|_| parser)
                        .map_err(|e| format!("Failed to set parser for {}: {}", stringify!($variant), e))
                }
            }
        )+

        #[derive(Debug, Clone)]
        pub enum Languages {
            $(
                $variant($type),
            )+
        }

        impl std::str::FromStr for Languages {
            type Err = String;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                let s = s.to_lowercase();
                let candidates = vec![
                    $(
                        (&$type::default().get_variant_names()[..], Languages::$variant($type::default())),
                    )+
                ];
                for (aliases, lang) in candidates {
                    if aliases.contains(&s.as_str()) {
                        return Ok(lang);
                    }
                }
                Err(format!("Unknown language: {}", s))
            }
        }

        impl Language for Languages {
            fn get_display_name(&self) -> &'static str {
                match self {
                    $(Languages::$variant(lang) => lang.get_display_name(),)+
                }
            }

            fn get_variant_names(&self) -> &'static [&'static str] {
                match self {
                    $(Languages::$variant(lang) => lang.get_variant_names(),)+
                }
            }

            fn get_parser(&self) -> Result<TSParser, String> {
                match self {
                    $(Languages::$variant(lang) => lang.get_parser(),)+
                }
            }
        }
    };
}

// --- Language Definitions --- //

define_languages! {
    Python => PyLang, ["python", "py", "python3"], tree_sitter_python::LANGUAGE;
    Java   => JavaLang, ["java", "javac", "jvm"], tree_sitter_java::LANGUAGE;
    C      => CLang, ["c", "h", "c++", "cpp", "clang"], tree_sitter_c::LANGUAGE;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_define_language() {
        let lang: Languages = "python".parse().unwrap();
        assert_eq!(lang.get_display_name(), "Python");
        assert!(lang.get_parser().is_ok());

        let lang: Languages = "java".parse().unwrap();
        assert_eq!(lang.get_display_name(), "Java");
        assert!(lang.get_parser().is_ok());

        let lang: Languages = "c".parse().unwrap();
        assert_eq!(lang.get_display_name(), "C");
        assert!(lang.get_parser().is_ok());
    }
}
