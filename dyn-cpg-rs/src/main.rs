use std::{fs::File, io::Read};

use clap::Parser as ClapParser;
// use gremlin_client::GremlinClient;
use tracing::{debug, error, info};

mod cli;
use cli::Cli;

use dyn_cpg_rs::logging;

// --- Helper Functions --- //

// use git2::{ObjectType, Repository};
// fn parse_commit(commit: &str) -> Result<String, String> {
//     let repo =
//         Repository::discover(".").map_err(|e| format!("Failed to discover repository: {}", e))?;

//     let object = repo
//         .revparse_single(commit)
//         .map_err(|e| format!("Failed to resolve commit '{}': {}", commit, e))?;

//     if object.kind() != Some(ObjectType::Commit) {
//         return Err(format!("Object '{}' is not a commit", commit));
//     }

//     Ok(object.id().to_string())
// }

// --- Main Entry Point --- //

fn main() -> Result<(), Box<dyn std::error::Error>> {
    logging::init();

    let args: Cli = Cli::parse();

    // DB
    // let client = GremlinClient::connect(args.db).map_err(|e| {
    //     error!("Failed to connect to Gremlin server: {}", e);
    //     e
    // })?;
    info!("Connected to Gremlin server");

    // Lang + Parser
    let mut parser = args.lang.get_parser().map_err(|e| {
        error!("Error initializing parser: {}", e);
        e
    })?;

    // Files
    let files: Vec<String> = args.files.into_iter().flat_map(|v| v).collect();

    let _old_files: Vec<String> = match (args.old_files, args.old_commit) {
        (Some(o_fs), None) => o_fs.into_iter().flat_map(|v| v).collect(),
        (None, Some(_commit)) => vec![], // TODO
        _ => return Err("Invalid combination of old_files and old_commit".into()),
    };

    for file_path in &files {
        debug!("Processing file: {}", file_path);

        let mut file = File::open(file_path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;

        // Parse test
        let tree = parser.parse(contents, None).ok_or_else(|| {
            error!("Failed to parse the code");
            "Parser returned None"
        })?;

        debug!("Parsed tree");

        // Convert tree to CPG
        let _cpg = args.lang.cst_to_cpg(tree).map_err(|e| {
            error!("Failed to convert tree to CPG: {}", e);
            e
        })?;
        debug!("Converted tree to CPG");
    }

    Ok(())
}
