use clap::Parser as ClapParser;
use std::str::FromStr;
use strum::VariantNames;
use strum_macros::EnumVariantNames;
use tree_sitter::Parser as TSParser;
// use tree_sitter::Tree as TSTree;
use glob::glob;
use url::Url;

// --- CLI Argument Parsing --- //

#[derive(ClapParser, Debug)]
#[command(name = "dyn-cpg-rs")]
#[command(about = "Incremental CPG generator and update tool", long_about = None)]
struct Cli {
    /// Database URI (e.g. ws://localhost:8182)
    db: String,

    /// Language of the source code
    #[arg(value_parser = Language::from_str)]
    lang: Language,

    /// Files/globs to parse
    #[arg(required = true)]
    files: Vec<String>,
}


#[derive(Debug, Clone, EnumVariantNames)]
enum Language {
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
            _ => Err(format!("Unknown language '{}', expected one of {:?}", s, Language::VARIANTS)),
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

// --- Verification of User Input --- //

fn expand_globs(globs: &[String]) -> Result<Vec<String>, String> {
    let mut result = Vec::new();

    for pattern in globs {
        let matches = glob(pattern).map_err(|e| format!("Invalid glob '{}': {}", pattern, e))?;
        for entry in matches {
            let path = entry.map_err(|e| format!("Glob error in '{}': {}", pattern, e))?;
            result.push(path.display().to_string());
        }
    }

    if result.is_empty() {
        return Err("No matching files found".to_string());
    }

    Ok(result)
}


fn verify_db_uri(uri: &str) -> Result<(), String> {
    let parsed = Url::parse(uri).map_err(|e| format!("Invalid URI: {}", e))?;
    if parsed.scheme() != "ws" && parsed.scheme() != "wss" {
        return Err(format!("Invalid scheme '{}', expected ws:// or wss://", parsed.scheme()));
    }
    Ok(())
}

fn verify_file_path(path: &str) -> Result<(), String> {
    if path.is_empty() {
        return Err("File path cannot be empty".to_string());
    }

    if std::fs::metadata(path).is_err() {
        return Err(format!("Could not read file: {}", path));
    }

    Ok(())
}


// --- Main Entry Point --- //

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Cli::parse();

    println!("Database URI: {}", args.db);
    println!("Language: {:?}", args.lang); // lang is verified by construction
    println!("Files: {:?}", args.files);

    verify_db_uri(&args.db).map_err(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    })?;

    let files = expand_globs(&args.files).map_err(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    })?;

    if files.is_empty() {
        eprintln!("Error: No files found matching the provided patterns");
        std::process::exit(1);
    }
    
    for file in &files {
        verify_file_path(file).map_err(|e| {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        })?;
    }

    
    let mut parser = args.lang.get_parser().map_err(|e| {
        eprintln!("Error initializing parser: {}", e);
        std::process::exit(1);
    })?;
    

    let tree = parser.parse("int main() { return 0; }", None);

    match tree {
        Some(t) => {
            println!("Parsed tree: {:?}", t);
        }
        None => {
            eprintln!("Failed to parse the code");
            std::process::exit(1);
        }
    }


    Ok(())
}
