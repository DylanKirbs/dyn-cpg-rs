use super::{Cpg, ListenerType, NodeId};
use std::collections::HashMap;
use strum_macros::Display;

#[derive(Debug, Clone, Display, PartialEq, Eq, Hash)]
pub enum EdgeType {
    Unknown,

    // AST
    SyntaxChild,
    SyntaxSibling,

    // Control Flow
    ControlFlowEpsilon,
    ControlFlowTrue,
    ControlFlowFalse,

    // Program Dependence
    PDControlTrue,
    PDControlFalse,
    PDData(String), // Identifier for data dependencies

    // Listener
    Listener(ListenerType),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Edge {
    pub from: NodeId,
    pub to: NodeId,
    pub type_: EdgeType,
    pub properties: HashMap<String, String>, // As little as possible should be stored here
}

// --- Edge query --- //

pub struct EdgeQuery<'a> {
    from: Option<&'a NodeId>,
    to: Option<&'a NodeId>,
    type_: Option<&'a EdgeType>,
}

impl<'query> EdgeQuery<'query> {
    pub fn new() -> Self {
        Self {
            from: None,
            to: None,
            type_: None,
        }
    }

    pub fn from<'id>(mut self, from: &'id NodeId) -> Self
    where
        'id: 'query,
    {
        self.from = Some(from);
        self
    }

    pub fn to<'id>(mut self, to: &'id NodeId) -> Self
    where
        'id: 'query,
    {
        self.to = Some(to);
        self
    }

    pub fn edge_type(mut self, ty: &'query EdgeType) -> Self {
        self.type_ = Some(ty);
        self
    }

    pub fn query(self, graph: &'query Cpg) -> Vec<&'query Edge> {
        graph
            .edges
            .values()
            .filter(|edge| {
                self.from.is_none_or(|f| edge.from == *f)
                    && self.to.is_none_or(|t| edge.to == *t)
                    && self.type_.is_none_or(|t| &edge.type_ == t)
            })
            .collect()
    }
}

impl<'a> Default for EdgeQuery<'a> {
    fn default() -> Self {
        Self::new()
    }
}
