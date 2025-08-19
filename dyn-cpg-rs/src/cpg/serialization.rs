use super::{Cpg, Edge, EdgeType, NodeId, NodeType};
use std::collections::HashSet;
use std::fmt::Write;
use std::path::PathBuf;

impl EdgeType {
    fn colour(&self) -> &'static str {
        match self {
            EdgeType::Unknown => "black",
            EdgeType::SyntaxChild => "blue",
            EdgeType::SyntaxSibling => "green",
            EdgeType::ControlFlowEpsilon => "red",
            EdgeType::ControlFlowTrue => "orange",
            EdgeType::ControlFlowFalse => "purple",
            EdgeType::PDControlTrue => "cyan",
            EdgeType::PDControlFalse => "magenta",
            EdgeType::PDData(_) => "brown",
            EdgeType::Listener(_) => "gray",
        }
    }

    fn label(&self) -> String {
        format!("{:?}", self)
            .replace("EdgeType::", "")
            .replace('_', " ")
    }
}

impl NodeId {
    pub fn as_str(&self) -> String {
        format!("{:?}", self)
    }
}

impl NodeType {
    fn colour(&self) -> &'static str {
        match self {
            NodeType::Comment | NodeType::LanguageImplementation(_) => "lightgray",
            _ => "black",
        }
    }

    fn label(&self) -> String {
        format!("{:?}", self)
            .replace("NodeType::", "")
            .replace('_', " ")
    }
}

pub trait CpgSerializer {
    fn start(&mut self);
    fn on_node(&mut self, cpg: &Cpg, id: NodeId);
    fn on_edge(&mut self, cpg: &Cpg, edge: &Edge);
    fn finish(&mut self);
    fn into_string(self) -> String;
}

// ---

pub struct DotSerializer {
    buf: String,
    visited: HashSet<NodeId>,
}

impl DotSerializer {
    pub fn new() -> Self {
        Self {
            buf: String::new(),
            visited: HashSet::new(),
        }
    }
}

impl CpgSerializer for DotSerializer {
    fn start(&mut self) {
        writeln!(self.buf, "digraph CPG {{").unwrap();
        writeln!(self.buf, "  rankdir=TB;").unwrap();
        writeln!(self.buf, "  node [shape=box];").unwrap();
    }

    fn on_node(&mut self, cpg: &Cpg, id: NodeId) {
        if !self.visited.insert(id) {
            return;
        }
        let node = cpg.get_node_by_id(&id).unwrap();
        let id_str = id.as_str();
        let pos = cpg
            .spatial_index
            .get_range_from_node(&id)
            .map_or("unknown".to_string(), |(s, e)| format!("{s}-{e}"));

        let label = format!(
            "{} {} {} {}",
            node.type_.label(),
            pos,
            node.properties
                .get("raw_kind")
                .map(String::as_str)
                .unwrap_or("unknown"),
            node.properties
                .get("name")
                .map(String::as_str)
                .unwrap_or(""),
        )
        .replace('"', "\\\"");

        writeln!(
            self.buf,
            "  {} [label=\"{}\" color={}];",
            id_str,
            label,
            node.type_.colour()
        )
        .unwrap();
    }

    fn on_edge(&mut self, _cpg: &Cpg, edge: &Edge) {
        writeln!(
            self.buf,
            "  {} -> {} [label=\"{}\", color=\"{}\"];",
            edge.from.as_str(),
            edge.to.as_str(),
            edge.type_.label(),
            edge.type_.colour()
        )
        .unwrap();
    }

    fn finish(&mut self) {
        self.buf.push_str("}\n");
    }

    fn into_string(self) -> String {
        self.buf
    }
}

// ---
pub struct SexpSerializer {
    buf: String,
}

impl SexpSerializer {
    pub fn new() -> Self {
        Self {
            buf: "(cpg\n".into(),
        }
    }
}

impl CpgSerializer for SexpSerializer {
    fn start(&mut self) {}

    fn on_node(&mut self, cpg: &Cpg, id: NodeId) {
        let node = cpg.get_node_by_id(&id).unwrap();
        let id_str = id.as_str();
        writeln!(
            self.buf,
            "  (node {} {} {:?})",
            id_str,
            node.type_.label(),
            node.properties
        )
        .unwrap();
    }

    fn on_edge(&mut self, _cpg: &Cpg, edge: &Edge) {
        writeln!(
            self.buf,
            "  (edge {} {} {:?})",
            edge.from.as_str(),
            edge.to.as_str(),
            edge.type_
        )
        .unwrap();
    }

    fn finish(&mut self) {
        self.buf.push_str(")\n");
    }

    fn into_string(self) -> String {
        self.buf
    }
}

// ---

impl Cpg {
    pub fn serialize<S: CpgSerializer>(&self, mut serializer: S) -> String {
        serializer.start();

        if let Some(root) = self.get_root() {
            let mut stack = vec![root];
            let mut visited = std::collections::HashSet::new();

            while let Some(id) = stack.pop() {
                if !visited.insert(id) {
                    continue;
                }

                serializer.on_node(self, id);

                for edge in self.get_outgoing_edges(id) {
                    serializer.on_edge(self, edge);
                    stack.push(edge.to);
                }
            }
        }

        serializer.finish();
        serializer.into_string()
    }

    pub fn serialize_to_file<S: CpgSerializer, P: Into<PathBuf>>(
        &self,
        serializer: S,
        path: P,
    ) -> std::io::Result<()> {
        let serialized = self.serialize(serializer);
        std::fs::write(path.into(), serialized)
    }
}
