use clap::{ArgGroup, Parser as ClapParser};
use std::str::FromStr;
use strum::VariantNames;
use strum_macros::EnumVariantNames;
use tree_sitter::Parser as TSParser;
// use tree_sitter::Tree as TSTree;
use git2::{ObjectType, Repository};
use glob::glob;
use url::Url;

// --- CLI Argument Parsing --- //

#[derive(ClapParser, Debug)]
#[command(name = "dyn-cpg-rs")]
#[command(about = "Incremental CPG generator and update tool", long_about = None)]
#[command(group(ArgGroup::new("old_input").required(true).args(["old_files", "old_commit"])))]
struct Cli {
    /// Database URI (e.g. ws://localhost:8182)
    #[arg(long, value_parser = parse_db_uri)]
    db: Url,

    /// Language of the source code
    #[arg(long, value_parser = Language::from_str)]
    lang: Language,

    /// Files/globs to parse
    #[arg(long, num_args = 1.., value_parser = parse_glob)]
    files: Vec<Vec<String>>,

    /// Old version of files/globs to diff against
    #[arg(long, num_args = 1.., value_parser = parse_glob)]
    old_files: Option<Vec<Vec<String>>>,

    /// Git commit hash to extract old file versions from
    #[arg(long, value_parser = parse_commit)]
    old_commit: Option<String>,
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

// --- Verification of User Input --- //

fn parse_db_uri(uri: &str) -> Result<Url, String> {
    let parsed = Url::parse(uri).map_err(|e| format!("Invalid URI: {}", e))?;
    let host = parsed.host_str().ok_or("Missing host in URI")?.to_string();
    let port = parsed.port().ok_or("Missing port in URI")?.to_string();
    let scheme = parsed.scheme();
    if scheme != "ws" && scheme != "wss" {
        return Err(format!("Unsupported scheme: {}", scheme));
    }
    if host.is_empty() {
        return Err("Host cannot be empty".to_string());
    }
    if port.is_empty() {
        return Err("Port cannot be empty".to_string());
    }
    Ok(parsed)
}

fn parse_glob(pattern: &str) -> Result<Vec<String>, String> {
    let matches: Vec<_> = glob(pattern)
        .map_err(|e| format!("Invalid glob '{}': {}", pattern, e))?
        .filter_map(Result::ok)
        .map(|p| p.display().to_string())
        .collect();

    if matches.is_empty() {
        return Err(format!("No files matched pattern '{}'", pattern));
    }

    Ok(matches)
}

fn parse_commit(commit: &str) -> Result<String, String> {
    let repo =
        Repository::discover(".").map_err(|e| format!("Failed to discover repository: {}", e))?;

    let object = repo
        .revparse_single(commit)
        .map_err(|e| format!("Failed to resolve commit '{}': {}", commit, e))?;

    if object.kind() != Some(ObjectType::Commit) {
        return Err(format!("Object '{}' is not a commit", commit));
    }

    Ok(object.id().to_string())
}

// --- Main Entry Point --- //

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Cli::parse();

    println!("Database URI: {}", args.db);
    println!("Language: {:?}", args.lang); // lang is verified by construction

    // Flatten the Vec<Vec<String>> into Vec<String>
    let files: Vec<String> = args.files.into_iter().flat_map(|v| v).collect();
    println!("Files: {:?}", files);

    let old_files: Vec<String> = match (args.old_files, args.old_commit) {
        (Some(o_fs), None) => o_fs.into_iter().flat_map(|v| v).collect(),
        (None, Some(commit)) => vec![], // TODO
        _ => unreachable!("how did we get here?"),
    };

    println!("Old files: {:?}", old_files);

    let mut parser = args.lang.get_parser().map_err(|e| {
        eprintln!("Error initializing parser: {}", e);
        std::process::exit(1);
    })?;

    // Some temporary code to test the parser
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
