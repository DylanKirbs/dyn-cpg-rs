use super::{DescendantTraversal, NodeId};
use strum_macros::Display;

#[derive(Debug, Clone, Display, PartialEq, Eq, Hash)]
pub enum IdenType {
    READ,
    WRITE,
    /// Unknown (treated as both)
    UNKNOWN,
}

#[derive(Debug, Clone, Display, Default, PartialEq, Eq, Hash)]
pub enum NodeType {
    /// Unknown node type (this is the default, but should not be present in a finished CPG)
    #[default]
    Unknown,

    /// Language-implementation specific nodes
    LanguageImplementation(String),

    /// Represents an error in the source code
    Error(String),

    /// The root of the CPG, representing the entire translation unit (e.g., a file)
    TranslationUnit,

    /// A function definition or declaration
    Function {
        name_traversals: Vec<DescendantTraversal>,
    },

    /// An internal node used to connect all return statements within a function.
    FunctionReturn,

    /// An identifier (variable, etc.)
    Identifier {
        type_: IdenType,
    },

    /// A statement that can be executed
    Statement,

    /// An expression that can be evaluated
    Expression,

    /// A type definition or usage
    Type,

    /// A comment in the source code
    Comment,

    /// Conditional branching constructs
    Branch {
        condition: DescendantTraversal,
        then_branch: DescendantTraversal,
        else_branch: DescendantTraversal,
    },

    /// Loop constructs
    Loop {
        condition: DescendantTraversal,
        body: DescendantTraversal,
    },

    /// Control Flow
    CFBreak,
    CFContinue,

    /// Compound statement, e.g., a block of code enclosed in braces
    Block,

    /// A function call expression
    Call,

    /// A return statement in a function
    Return,
}

/// A node in the Code Property Graph (CPG)
#[derive(Debug, Clone, Default)]
pub struct Node {
    /// The readable name of the node (if it has one)
    pub name: Option<String>,

    /// The "raw" Tree-Sitter type of the node
    pub raw_type: String,

    /// Type of the node
    pub type_: NodeType,

    /// The offset of the node from it's left sibling's end (or it's parent's start if no left sibling)
    pub offset: usize,

    /// Size of the node in bytes (width in bytes)
    pub size: usize,

    /// DF Reads
    pub df_reads: Vec<String>,
    /// DF Writes
    pub df_writes: Vec<String>,
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.raw_type == other.raw_type
            && self.type_ == other.type_
            && self.offset == other.offset
            && self.size == other.size
    }
}

impl NodeId {
    pub fn as_str(&self) -> String {
        format!("\"{:?}\"", self)
            .replace("NodeId(", "")
            .replace(')', "")
    }
}

impl NodeType {
    pub fn colour(&self) -> &'static str {
        match self {
            NodeType::Comment | NodeType::LanguageImplementation(_) => "lightgray",
            _ => "black",
        }
    }

    pub fn label(&self) -> String {
        match self {
            NodeType::LanguageImplementation(lang) => format!("LangImpl({})", lang),
            NodeType::Error(msg) => format!("Error({})", msg),
            NodeType::TranslationUnit => "TranslationUnit".to_string(),
            NodeType::Function { .. } => "Function".to_string(),
            NodeType::FunctionReturn => "FunctionReturn".to_string(),
            NodeType::Identifier { type_ } => format!("Identifier({})", type_),
            NodeType::Statement => "Statement".to_string(),
            NodeType::Expression => "Expression".to_string(),
            NodeType::Type => "Type".to_string(),
            NodeType::Comment => "Comment".to_string(),
            NodeType::Branch { .. } => "Branch".to_string(),
            NodeType::Loop { .. } => "Loop".to_string(),
            NodeType::CFBreak => "CFBreak".to_string(),
            NodeType::CFContinue => "CFContinue".to_string(),
            NodeType::Block => "Block".to_string(),
            NodeType::Call => "Call".to_string(),
            NodeType::Return => "Return".to_string(),
            NodeType::Unknown => "UNKNOWN".to_string(),
        }
    }
}
