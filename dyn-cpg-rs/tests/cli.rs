use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

#[test]
fn test_db_scheme() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("dyn-cpg-rs")?;

    cmd.arg("http://localhost:2025").arg("c").arg("**/*.c");
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("Error: Invalid scheme \'http\', expected ws:// or wss://"));

    Ok(())
}
