use dyn_cpg_rs::{
    cpg::serialization::DotSerializer, languages::RegisteredLanguage, resource::Resource,
};

#[test]
fn test_construct_parsing() {
    let lang: RegisteredLanguage = "c".parse().expect("Failed to parse language");
    let mut parser = lang.get_parser().expect("Failed to get parser for C");

    let s =
        Resource::new("samples/constructs.c").expect("Failed to create resource for constructs.c");

    let src = s.read_bytes().expect("Failed to read constructs.c");

    let tree = parser.parse(&src, None).expect("Failed to parse sample1.c");

    let cpg = lang
        .cst_to_cpg(tree, src)
        .expect("Failed to convert tree to CPG");

    cpg.serialize_to_file(&mut DotSerializer::new(), "debug/constructs.dot", None)
        .expect("Failed to write constructs.dot");
}

// For each c file in samples/*.c generate the dot file
#[test]
fn test_all_samples() {
    let lang: RegisteredLanguage = "c".parse().expect("Failed to parse language");
    let mut parser = lang.get_parser().expect("Failed to get parser for C");

    let paths = std::fs::read_dir("samples").expect("Failed to read samples directory");

    for path in paths {
        let path = path.expect("Failed to read path").path();
        if path.extension().and_then(|s| s.to_str()) != Some("c") {
            continue;
        }

        let s = Resource::new(path.clone())
            .expect(&format!("Failed to create resource for {:?}", path));

        let src = s.read_bytes().expect(&format!("Failed to read {:?}", path));

        let tree = parser
            .parse(&src, None)
            .expect(&format!("Failed to parse {:?}", path));

        let cpg = lang
            .cst_to_cpg(tree, src)
            .expect(&format!("Failed to convert tree to CPG for {:?}", path));

        let dot_path = format!(
            "debug/{}.dot",
            path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
        );

        cpg.serialize_to_file(&mut DotSerializer::new(), &dot_path, None)
            .expect(&format!("Failed to write {:?}", dot_path));
    }
}
