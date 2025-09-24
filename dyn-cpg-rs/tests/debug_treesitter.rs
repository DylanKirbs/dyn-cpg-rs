use tree_sitter::{InputEdit, Point};

#[test]  
fn test_treesitter_incremental_parsing() {
    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&tree_sitter_c::LANGUAGE.into()).unwrap();
    
    // Original source
    let source1 = b"int main() { return 0; }";
    let source2 = b"int main() { int x = 5; return x; }";
    
    // Parse original
    let mut tree = parser.parse(source1, None).unwrap();
    println!("=== ORIGINAL TREE ===");
    print_tree(&tree.root_node(), source1, 0);
    
    // Apply edits for incremental parsing
    let edits = vec![
        InputEdit {
            start_byte: 13,
            old_end_byte: 13,
            new_end_byte: 24,
            start_position: Point { row: 0, column: 13 },
            old_end_position: Point { row: 0, column: 13 },
            new_end_position: Point { row: 0, column: 24 },
        },
        InputEdit {
            start_byte: 31,  // Corrected: 20 + 11 chars inserted = 31
            old_end_byte: 33, // Corrected: 22 + 11 chars inserted = 33  
            new_end_byte: 33, // Same in new source
            start_position: Point { row: 0, column: 31 },
            old_end_position: Point { row: 0, column: 33 },
            new_end_position: Point { row: 0, column: 33 },
        },
    ];
    
    for edit in edits {
        tree.edit(&edit);
    }
    
    // Parse incrementally
    let new_tree = parser.parse(source2, Some(&tree)).unwrap();
    println!("\n=== INCREMENTAL TREE ===");
    print_tree(&new_tree.root_node(), source2, 0);
    
    // Parse from scratch for comparison
    let ref_tree = parser.parse(source2, None).unwrap();
    println!("\n=== REFERENCE TREE ===");
    print_tree(&ref_tree.root_node(), source2, 0);
}

fn print_tree(node: &tree_sitter::Node, source: &[u8], depth: usize) {
    let indent = "  ".repeat(depth);
    let text = &source[node.start_byte()..node.end_byte()];
    let text_str = String::from_utf8_lossy(text);
    
    println!("{}({}) [{}..{}] '{}'", 
             indent, node.kind(), node.start_byte(), node.end_byte(), text_str);
    
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i) {
            print_tree(&child, source, depth + 1);
        }
    }
}
