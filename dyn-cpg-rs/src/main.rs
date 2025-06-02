use clap::Parser as ClapParser;
use tracing::{debug, error, info};

mod cli;
use cli::Cli;

use dyn_cpg_rs::{logging, resource::Resource};

// --- Main Entry Point --- //

/*
|              | File in DB   | File not in DB |
| File exists  | Update       | Insert         |
| File missing | Delete (ask) | N/A            |

If we can't parse the old file for any reason, we can do a full CPG rebuild.
*/

fn main() -> Result<(), Box<dyn std::error::Error>> {
    logging::init();

    let args: Cli = Cli::parse();

    // DB
    // let client = gremlin_client::GremlinClient::connect(args.db).map_err(|e| {
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
    let mut files: Vec<Resource> = args
        .files
        .into_iter()
        .flat_map(|v| v)
        .map(|file| Resource::new(file))
        .collect();

    for file_resource in &mut files {
        debug!("Processing file: {:?}", file_resource.raw_path());

        let content = file_resource.read_bytes().map_err(|e| {
            error!(
                "Failed to read file {}: {:?}",
                file_resource.raw_path().display(),
                e
            );
            format!("{:?}", e)
        })?;

        let tree = parser.parse(content, None).ok_or_else(|| {
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
