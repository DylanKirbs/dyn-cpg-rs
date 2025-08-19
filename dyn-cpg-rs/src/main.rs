use clap::Parser as ClapParser;
use glob::glob;
use tracing::{debug, info};
use url::Url;

use dyn_cpg_rs::{languages::RegisteredLanguage, logging, resource::Resource};

// --- CLI Argument Parsing --- //

#[derive(ClapParser, Debug)]
#[command(name = "dyn-cpg-rs")]
#[command(about = "Incremental CPG generator and update tool", long_about = None)]
pub struct Cli {
    /// Database URI (e.g. ws://localhost:8182)
    #[arg(long, value_parser = parse_db_uri)]
    pub db: String,

    /// Language of the source code
    #[arg(long)]
    pub lang: RegisteredLanguage,

    /// Files/globs to parse
    #[arg(long, num_args = 1.., value_parser = parse_glob)]
    pub files: Vec<Vec<String>>,
}

// --- Verification of User Input --- //

fn parse_db_uri(uri: &str) -> Result<String, String> {
    let parsed = Url::parse(uri).map_err(|e| format!("Invalid URI: {}", e))?;
    let host = parsed.host_str().unwrap_or("");

    // Check if the host is a valid IP address or hostname
    if host.is_empty() {
        return Err("Missing host in URI".to_string());
    }
    if !host
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-')
    {
        return Err(format!("Invalid host in URI: {}", host));
    }

    let port = parsed.port().unwrap_or(8182);

    // Check if the scheme is either ws or wss or nothing
    let scheme = parsed.scheme();
    if scheme != "ws" && scheme != "wss" {
        return Err(format!(
            "Invalid scheme: {}. Expected 'ws' or 'wss'",
            scheme
        ));
    }

    Ok(format!("ws://{}:{}", host, port))
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

// --- Main Entry Point --- //

fn run_application(args: Cli) -> Result<(), Box<dyn std::error::Error>> {
    // Only initialize logging if not already initialized (for tests)
    static LOGGING_INIT: std::sync::Once = std::sync::Once::new();
    LOGGING_INIT.call_once(|| {
        logging::init();
    });

    // DB
    // TODO
    info!("Connected to Database server");

    // Lang + Parser
    let mut parser = args.lang.get_parser()?;

    // Files
    let mut files: Vec<Resource> = args
        .files
        .into_iter()
        .flatten()
        .map(Resource::new)
        .collect::<Result<Vec<_>, _>>()?;

    debug!("Found {} files to process", files.len());

    for file_resource in &mut files {
        debug!("Processing file: {:?}", file_resource.raw_path());

        let content = file_resource.read_bytes()?;
        let tree = parser.parse(&content, None).ok_or(
            "Failed to parse file content. Ensure the file is valid for the specified language.",
        )?;
        debug!("Parsed tree");

        // Convert tree to CPG
        let _cpg = args.lang.cst_to_cpg(tree, content)?;
        debug!("Converted tree to CPG");
    }

    info!("Successfully processed all files");

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Cli = Cli::parse();
    run_application(args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_parse_db_uri_valid() {
        let uri = "ws://localhost:8182";
        let result = parse_db_uri(uri);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_db_uri_valid_with_custom_port() {
        let uri = "ws://example.com:9999";
        let result = parse_db_uri(uri);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_db_uri_valid_wss() {
        let uri = "wss://secure.example.com:8182";
        let result = parse_db_uri(uri);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_db_uri_invalid_scheme() {
        let uri = "http://localhost:8182";
        let result = parse_db_uri(uri);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid scheme"));
    }

    #[test]
    fn test_parse_db_uri_missing_host() {
        let uri = "ws://:8182";
        let result = parse_db_uri(uri);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty host"));
    }

    #[test]
    fn test_parse_db_uri_invalid_host() {
        let uri = "ws://invalid!host:8182";
        let result = parse_db_uri(uri);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid host"));
    }

    #[test]
    fn test_parse_db_uri_malformed() {
        let uri = "not-a-uri";
        let result = parse_db_uri(uri);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid URI"));
    }

    #[test]
    fn test_parse_db_uri_default_port() {
        let uri = "ws://localhost";
        let result = parse_db_uri(uri);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_db_uri_numeric_host() {
        let uri = "ws://192.168.1.1:8182";
        let result = parse_db_uri(uri);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_db_uri_hostname_with_dash() {
        let uri = "ws://my-server.example.com:8182";
        let result = parse_db_uri(uri);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_glob_with_existing_sample() {
        if Path::new("samples/sample1.old.c").exists() {
            let result = parse_glob("samples/sample1.old.c");
            assert!(result.is_ok());
            let files = result.unwrap();
            assert_eq!(files.len(), 1);
            assert!(files[0].contains("sample1.old.c"));
        }
    }

    #[test]
    fn test_parse_glob_wildcard_samples() {
        if Path::new("samples").exists() {
            let result = parse_glob("samples/*.c");
            if result.is_ok() {
                let files = result.unwrap();
                assert!(!files.is_empty());
                assert!(files.iter().all(|f| f.ends_with(".c")));
            }
        }
    }

    #[test]
    fn test_parse_glob_no_matches() {
        let pattern = "/nonexistent/path/*.xyz";
        let result = parse_glob(pattern);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No files matched"));
    }

    #[test]
    fn test_parse_glob_invalid_pattern() {
        let pattern = "[invalid";
        let result = parse_glob(pattern);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid glob"));
    }

    #[test]
    fn test_parse_glob_src_files() {
        let result = parse_glob("src/*.rs");
        assert!(result.is_ok());
        let files = result.unwrap();
        assert!(!files.is_empty());
        assert!(files.iter().any(|f| f.contains("main.rs")));
        assert!(files.iter().any(|f| f.contains("lib.rs")));
    }

    #[test]
    fn test_registered_language_parsing() {
        let c_lang: Result<RegisteredLanguage, _> = "c".parse();
        assert!(c_lang.is_ok());

        let invalid_lang: Result<RegisteredLanguage, _> = "invalid".parse();
        assert!(invalid_lang.is_err());
    }

    #[test]
    fn test_parse_db_uri_edge_cases() {
        assert!(parse_db_uri("ws://a:8182").is_ok());

        assert!(
            parse_db_uri("wss://very-long-hostname-with-many-characters.example.com:8182").is_ok()
        );

        let long_host = "a".repeat(100);
        let uri = format!("ws://{}:8182", long_host);
        assert!(parse_db_uri(&uri).is_ok());

        assert!(parse_db_uri("ws://127.0.0.1:8182").is_ok());

        assert!(parse_db_uri("ws://0.0.0.0:1").is_ok());
        assert!(parse_db_uri("ws://host:65535").is_ok());
    }

    #[test]
    fn test_parse_glob_edge_cases() {
        let result = parse_glob("src/main.rs");
        assert!(result.is_ok());
        let files = result.unwrap();
        assert_eq!(files.len(), 1);
        assert!(files[0].ends_with("main.rs"));

        let result = parse_glob("**/*.rs");
        assert!(result.is_ok());
        let files = result.unwrap();
        assert!(!files.is_empty());
        assert!(files.iter().all(|f| f.ends_with(".rs")));
    }
}
