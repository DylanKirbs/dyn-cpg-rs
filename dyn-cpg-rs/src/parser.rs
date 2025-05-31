use tree_sitter::Parser as TSParser;

// --- The Language Trait and Construction Macro --- //

pub trait Language: std::fmt::Debug + Clone {
    /// Get the display name of the language.
    fn get_display_name(&self) -> &'static str;

    /// Get the variant names and extensions for the language.
    fn get_variant_names(&self) -> &'static [&'static str];

    /// Get a Tree-sitter parser for the language.
    /// Each call to this function should return a new parser instance.
    fn get_parser(&self) -> Result<TSParser, String>;
}

macro_rules! define_RegisteredLanguage {
    (
        $(
            $variant:ident => [$($alias:expr),+], $lang:path;
        )+
    ) => {
        $(
            #[derive(Debug, Clone)]
            pub struct $variant;

            impl Default for $variant {
                fn default() -> Self {
                    Self
                }
            }

            impl Language for $variant {
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
                        (&$variant::default().get_variant_names()[..], RegisteredLanguage::$variant($variant::default())),
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

        impl Language for RegisteredLanguage {
            fn get_display_name(&self) -> &'static str {
                match self {
                    $(RegisteredLanguage::$variant(lang) => lang.get_display_name(),)+
                }
            }

            fn get_variant_names(&self) -> &'static [&'static str] {
                match self {
                    $(RegisteredLanguage::$variant(lang) => lang.get_variant_names(),)+
                }
            }

            fn get_parser(&self) -> Result<TSParser, String> {
                match self {
                    $(RegisteredLanguage::$variant(lang) => lang.get_parser(),)+
                }
            }
        }
    };
}

// --- Language Definitions --- //

define_RegisteredLanguage! {
    Python => ["python", "py", "python3"], tree_sitter_python::LANGUAGE;
    Java   => ["java", "javac", "jvm"], tree_sitter_java::LANGUAGE;
    C      => ["c", "h", "c++", "cpp", "clang"], tree_sitter_c::LANGUAGE;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_define_language() {
        let lang: RegisteredLanguage = "python".parse().unwrap();
        assert_eq!(lang.get_display_name(), "Python");
        assert!(lang.get_parser().is_ok());

        let lang: RegisteredLanguage = "java".parse().unwrap();
        assert_eq!(lang.get_display_name(), "Java");
        assert!(lang.get_parser().is_ok());

        let lang: RegisteredLanguage = "c".parse().unwrap();
        assert_eq!(lang.get_display_name(), "C");
        assert!(lang.get_parser().is_ok());
    }
}
