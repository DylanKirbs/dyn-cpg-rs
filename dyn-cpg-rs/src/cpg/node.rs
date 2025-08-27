use super::DescendantTraversal;
use std::collections::HashMap;
use strum_macros::Display;

#[derive(Debug, Clone, Display, PartialEq, Eq, Hash)]
pub enum IdenType {
    READ,
    WRITE,
    /// Unknown (treated as both)
    UNKNOWN,
}

#[derive(Debug, Clone, Display, PartialEq, Eq, Hash)]
pub enum NodeType {
    /// Language-implementation specific nodes
    LanguageImplementation(String),

    /// Represents an error in the source code
    Error(String),

    /// The root of the CPG, representing the entire translation unit (e.g., a file)
    TranslationUnit,

    /// A function definition or declaration
    Function {
        name_traversal: DescendantTraversal,
        /// Optional name, can be derived from the traversal
        name: Option<String>,
    },
    /// An internal node used to connect all return statements within a function.
    FunctionReturn,

    /// An identifier (variable, etc.)
    Identifier {
        type_: IdenType,
        name: Option<String>,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    pub type_: NodeType,
    pub properties: HashMap<String, String>, // As little as possible should be stored here
}

impl Node {
    pub fn update_type(&mut self, type_: NodeType) -> Option<()> {
        if self.type_ != type_ {
            self.type_ = type_;
            Some(())
        } else {
            None
        }
    }
}
