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

    cpg.serialize_to_file(&mut DotSerializer::new(), "constructs.dot", None)
        .expect("Failed to write constructs.dot");
}
