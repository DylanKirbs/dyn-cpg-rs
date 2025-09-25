use super::{Cpg, Edge, EdgeId, NodeId};
use crate::cpg::spatial_index::SpatialIndex;
use std::collections::HashSet;
use std::fmt::Write;
use std::path::PathBuf;
use strum::Display;
use thiserror::Error;

#[derive(Debug, Error, Display)]
pub enum SerializationError {
    BadRequest(String),
    NodeNotFound(NodeId),
    EdgeNotFound(EdgeId),
    FmtError(#[from] std::fmt::Error),
    IoError(#[from] std::io::Error),
}

pub trait CpgSerializer<T> {
    fn serialize(&mut self, cpg: &Cpg, root: Option<NodeId>) -> Result<T, SerializationError>;
}

// --- DOT --- //

pub struct DotSerializer {
    buf: String,
    visited: HashSet<NodeId>,

    id_map: std::collections::HashMap<NodeId, usize>,
    next_id: usize,
}

impl Default for DotSerializer {
    fn default() -> Self {
        Self::new()
    }
}

impl DotSerializer {
    pub fn new() -> Self {
        Self {
            buf: String::new(),
            visited: HashSet::new(),
            id_map: std::collections::HashMap::new(),
            next_id: 0,
        }
    }

    fn start(&mut self, _cpg: &Cpg) -> Result<(), SerializationError> {
        writeln!(self.buf, "digraph CPG {{")?;
        writeln!(self.buf, "  rankdir=TB;")?;
        writeln!(self.buf, "  node [shape=box];")?;
        Ok(())
    }

    fn on_node(&mut self, cpg: &Cpg, id: NodeId) -> Result<(), SerializationError> {
        if !self.visited.insert(id) {
            return Ok(());
        }

        let canon = self.id_map.entry(id).or_insert_with(|| {
            let id = self.next_id;
            self.next_id += 1;
            id
        });

        let node = cpg
            .get_node_by_id(&id)
            .ok_or(SerializationError::NodeNotFound(id))?;
        let id_str = id.as_str();
        let pos = cpg
            .spatial_index
            .get_node_span(id)
            .map_or("unknown".to_string(), |(s, e)| format!("{s}-{e}"));

        let label = format!(
            "canonical-id:{} type:{} pos:{} kind:{} name:{} id:{}",
            canon,
            node.type_.label(),
            pos,
            node.raw_type,
            node.name.as_ref().unwrap_or(&"Unnamed".to_string()),
            id_str,
        )
        .replace('"', "\\\"");

        writeln!(
            self.buf,
            "  {} [label=\"{}\" color={}];",
            id_str,
            label,
            node.type_.colour()
        )?;

        Ok(())
    }

    fn on_edge(&mut self, _cpg: &Cpg, edge: &Edge) -> Result<(), SerializationError> {
        writeln!(
            self.buf,
            "  {} -> {} [label=\"{}\", color=\"{}\"];",
            edge.from.as_str(),
            edge.to.as_str(),
            edge.type_.label(),
            edge.type_.colour()
        )?;

        Ok(())
    }

    fn walk(&mut self, cpg: &Cpg, root: NodeId) -> Result<(), SerializationError> {
        let mut stack = vec![root];
        let mut visited = HashSet::new();

        while let Some(id) = stack.pop() {
            if !visited.insert(id) {
                continue;
            }
            self.on_node(cpg, id)?;
            for edge in cpg.get_deterministic_sorted_outgoing_edges(id) {
                stack.push(edge.to);
                self.on_edge(cpg, edge)?;
            }
        }

        Ok(())
    }

    fn finish(&mut self, _cpg: &Cpg) {
        self.buf.push_str("}\n");
    }

    fn write(&mut self) -> String {
        self.buf.clone()
    }
}

impl CpgSerializer<String> for DotSerializer {
    fn serialize(&mut self, cpg: &Cpg, root: Option<NodeId>) -> Result<String, SerializationError> {
        self.start(cpg)?;
        if let Some(root) = root {
            self.walk(cpg, root)?;
        } else if let Some(root) = cpg.get_root() {
            self.walk(cpg, root)?;
        } else {
            Err(SerializationError::BadRequest(
                "Invalid or missing root".to_string(),
            ))?;
        }
        self.finish(cpg);
        Ok(self.write())
    }
}

// --- S-EXPR --- //

pub struct SexpSerializer {
    buf: String,
    include_common_props: bool,

    id_map: std::collections::HashMap<NodeId, usize>,
    next_id: usize,
}

impl Default for SexpSerializer {
    fn default() -> Self {
        Self::new()
    }
}

impl SexpSerializer {
    pub fn new() -> Self {
        Self {
            buf: String::new(),
            include_common_props: true,
            id_map: std::collections::HashMap::new(),
            next_id: 1,
        }
    }

    fn on_node_enter(&mut self, cpg: &Cpg, id: NodeId) -> Result<(), SerializationError> {
        let node = cpg
            .get_node_by_id(&id)
            .ok_or(SerializationError::NodeNotFound(id))?;
        // Assign a canonical id for serialization (stable per-serialization)
        let canon = if let Some(&n) = self.id_map.get(&id) {
            n
        } else {
            let n = self.next_id;
            self.id_map.insert(id, n);
            self.next_id += 1;
            n
        };
        write!(
            self.buf,
            "({} :canonical-id \"{}\" :id {} :span \"{:?}\"",
            node.type_.label(),
            canon,
            id.as_str(),
            cpg.get_node_offsets_by_id(&id)
                .unwrap_or((usize::MAX, usize::MAX))
        )?;

        if self.include_common_props {
            let name = node.name.as_ref();
            let kind = node.raw_type.as_str();
            match name {
                Some(n) => write!(
                    self.buf,
                    " :name \"{}\" :kind \"{}\"",
                    escape_sexp(n),
                    escape_sexp(kind)
                ),
                None => write!(self.buf, " :kind \"{}\"", escape_sexp(kind)),
            }?;
        }

        Ok(())
    }

    fn on_loop_enter(&mut self, cpg: &Cpg, id: NodeId) -> Result<(), SerializationError> {
        let node = cpg
            .get_node_by_id(&id)
            .ok_or(SerializationError::NodeNotFound(id))?;
        write!(self.buf, "{} [visited {}]", node.type_.label(), id.as_str())?;
        Ok(())
    }

    fn walk_recursive(
        &mut self,
        cpg: &Cpg,
        id: NodeId,
        visiting: &mut HashSet<NodeId>,
        visited: &mut HashSet<NodeId>,
        depth: usize,
    ) -> Result<(), SerializationError> {
        if visited.contains(&id) {
            // Already serialized this node elsewhere → just emit a loop marker
            write!(self.buf, "(")?;
            self.on_loop_enter(cpg, id)?;
            write!(self.buf, ")")?;
            return Ok(());
        }

        if !visiting.insert(id) {
            // We're re-entering a node already on the current path → cycle
            write!(self.buf, "(")?;
            self.on_loop_enter(cpg, id)?;
            write!(self.buf, ")")?;
            return Ok(());
        }

        self.on_node_enter(cpg, id)?;

        for edge in cpg.get_deterministic_sorted_outgoing_edges(id) {
            write!(
                self.buf,
                "\n{}(-> {} ",
                "  ".repeat(depth),
                edge.type_.label()
            )?;

            if visiting.contains(&edge.to) {
                self.on_loop_enter(cpg, edge.to)?;
            } else {
                self.walk_recursive(cpg, edge.to, visiting, visited, depth + 1)?;
            }

            write!(self.buf, ")")?;
        }

        write!(self.buf, ")")?;

        visiting.remove(&id);
        visited.insert(id);

        Ok(())
    }

    fn start(&mut self, _cpg: &Cpg) {
        self.buf.push_str("(cpg ");
    }

    fn finish(&mut self, _cpg: &Cpg) {
        self.buf.push_str(")\n");
    }

    fn write(&mut self) -> String {
        self.buf.clone()
    }
}

impl CpgSerializer<String> for SexpSerializer {
    fn serialize(&mut self, cpg: &Cpg, root: Option<NodeId>) -> Result<String, SerializationError> {
        self.start(cpg);
        if let Some(root) = root {
            let mut visiting = HashSet::new();
            let mut visited = HashSet::new();
            self.walk_recursive(cpg, root, &mut visiting, &mut visited, 1)?;
        } else if let Some(root) = cpg.get_root() {
            let mut visiting = HashSet::new();
            let mut visited = HashSet::new();
            self.walk_recursive(cpg, root, &mut visiting, &mut visited, 1)?;
        } else {
            Err(SerializationError::BadRequest(
                "Invalid or missing root".to_string(),
            ))?;
        }
        self.finish(cpg);
        Ok(self.write())
    }
}

// --- Neo4j --- //

// TODO implement a Neo4j serializer

// --- CPG --- //

impl Cpg {
    pub fn serialize<T, S: CpgSerializer<T>>(
        &self,
        serializer: &mut S,
        root: Option<NodeId>,
    ) -> Result<T, SerializationError> {
        serializer.serialize(self, root)
    }

    pub fn serialize_to_file<T: AsRef<[u8]>, S: CpgSerializer<T>, P: Into<PathBuf>>(
        &self,
        serializer: &mut S,
        path: P,
        root: Option<NodeId>,
    ) -> Result<(), SerializationError> {
        std::fs::write(path.into(), serializer.serialize(self, root)?)?;
        Ok(())
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
