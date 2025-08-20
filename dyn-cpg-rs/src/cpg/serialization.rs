use super::{Cpg, Edge, EdgeType, NodeId, NodeType};
use std::collections::HashSet;
use std::fmt::Write;
use std::path::PathBuf;

pub trait CpgSerializer {
    fn serialize(&mut self, cpg: &Cpg) -> String;
}

// --- DOT --- //

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

    fn start(&mut self, _cpg: &Cpg) {
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

    fn walk(&mut self, cpg: &Cpg, root: NodeId) {
        let mut stack = vec![root];
        let mut visited = HashSet::new();

        while let Some(id) = stack.pop() {
            if !visited.insert(id) {
                continue;
            }
            self.on_node(cpg, id);
            for edge in cpg.get_outgoing_edges(id) {
                stack.push(edge.to);
                self.on_edge(cpg, edge);
            }
        }
    }

    fn finish(&mut self, _cpg: &Cpg) {
        self.buf.push_str("}\n");
    }

    fn into_string(&mut self) -> String {
        self.buf.clone()
    }
}

impl CpgSerializer for DotSerializer {
    fn serialize(&mut self, cpg: &Cpg) -> String {
        self.start(cpg);
        if let Some(root) = cpg.get_root() {
            self.walk(cpg, root);
        }
        self.finish(cpg);
        self.into_string()
    }
}

// --- S-EXPR --- //

pub struct SexpSerializer {
    buf: String,
    include_common_props: bool,
}

impl SexpSerializer {
    pub fn new() -> Self {
        Self {
            buf: String::new(),
            include_common_props: true,
        }
    }

    fn on_node_enter(&mut self, cpg: &Cpg, id: NodeId) {
        let node = cpg.get_node_by_id(&id).unwrap();
        write!(self.buf, "({}", node.type_.label()).unwrap();

        if self.include_common_props {
            let name = node.properties.get("name").map(String::as_str);
            let kind = node.properties.get("raw_kind").map(String::as_str);
            match (name, kind) {
                (Some(n), Some(k)) => {
                    write!(self.buf, " :name \"{}\" :kind {}", escape_sexp(n), k).unwrap()
                }
                (Some(n), None) => write!(self.buf, " :name \"{}\"", escape_sexp(n)).unwrap(),
                (None, Some(k)) => write!(self.buf, " :kind {}", k).unwrap(),
                (None, None) => {}
            }
        }
    }

    fn on_loop_enter(&mut self, cpg: &Cpg, id: NodeId) {
        let node = cpg.get_node_by_id(&id).unwrap();
        write!(self.buf, "{} [loop]", node.type_.label()).unwrap();
    }

    fn walk_recursive(&mut self, cpg: &Cpg, id: NodeId, visiting: &mut HashSet<NodeId>) {
        if !visiting.insert(id) {
            // This branch means we're re-entering a node that's on the current path -> cycle.
            // Emit a compact loop mention (node type + [loop]).
            write!(self.buf, "(").unwrap();
            self.on_loop_enter(cpg, id);
            write!(self.buf, ")").unwrap();
            return;
        }

        self.on_node_enter(cpg, id);

        for edge in cpg.get_outgoing_edges(id) {
            write!(self.buf, " (-> {} ", edge.type_.label()).unwrap();

            if visiting.contains(&edge.to) {
                self.on_loop_enter(cpg, edge.to);
            } else {
                self.walk_recursive(cpg, edge.to, visiting);
            }

            write!(self.buf, ")").unwrap();
        }

        write!(self.buf, ")").unwrap();
        visiting.remove(&id);
    }

    fn start(&mut self, _cpg: &Cpg) {
        self.buf.push_str("(cpg ");
    }

    fn finish(&mut self, _cpg: &Cpg) {
        self.buf.push_str(")\n");
    }

    fn into_string(&mut self) -> String {
        self.buf.clone()
    }
}

impl CpgSerializer for SexpSerializer {
    fn serialize(&mut self, cpg: &Cpg) -> String {
        self.start(cpg);
        if let Some(root) = cpg.get_root() {
            let mut visiting = HashSet::new();
            self.walk_recursive(cpg, root, &mut visiting);
        }
        self.finish(cpg);
        self.into_string()
    }
}

// --- CPG --- //

impl Cpg {
    pub fn serialize<S: CpgSerializer>(&self, serializer: &mut S) -> String {
        serializer.serialize(self)
    }

    pub fn serialize_to_file<S: CpgSerializer, P: Into<PathBuf>>(
        &self,
        serializer: &mut S,
        path: P,
    ) -> std::io::Result<()> {
        std::fs::write(path.into(), serializer.serialize(self))
    }
}

// --- helpers --- //

fn escape_sexp(s: &str) -> String {
    // Minimal escaping for double quotes and backslashes for inline strings.
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            _ => out.push(ch),
        }
    }
    out
}

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
