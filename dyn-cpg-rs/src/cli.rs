use clap::Parser as ClapParser;
use glob::glob;
use gremlin_client::ConnectionOptions;
use url::Url;

use dyn_cpg_rs::languages::RegisteredLanguage;

// --- CLI Argument Parsing --- //

#[derive(ClapParser, Debug)]
#[command(name = "dyn-cpg-rs")]
#[command(about = "Incremental CPG generator and update tool", long_about = None)]
pub struct Cli {
    /// Database URI (e.g. ws://localhost:8182)
    #[arg(long, value_parser = parse_db_uri)]
    pub db: ConnectionOptions,

    /// Language of the source code
    #[arg(long)]
    pub lang: RegisteredLanguage,

    /// Files/globs to parse
    #[arg(long, num_args = 1.., value_parser = parse_glob)]
    pub files: Vec<Vec<String>>,
}

// --- Verification of User Input --- //

fn parse_db_uri(uri: &str) -> Result<ConnectionOptions, String> {
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

    Ok(ConnectionOptions::builder().host(host).port(port).build())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_db_uri() {
        assert!(parse_db_uri("ws://localhost:8182").is_ok());
        assert!(parse_db_uri("wss://example.com:1234").is_ok());
        assert!(parse_db_uri("http://invalid.com").is_err());
        assert!(parse_db_uri("ws://").is_err());
    }

    #[test]
    fn test_parse_glob() {
        assert!(parse_glob("src/**/*.rs").is_ok());
        assert!(parse_glob("nonexistent/*.rs").is_err());
    }
}
