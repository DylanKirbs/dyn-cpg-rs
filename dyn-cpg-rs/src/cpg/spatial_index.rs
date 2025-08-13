use super::NodeId;
use std::collections::{BTreeMap, HashMap};

#[derive(Debug, Clone, Default)]
pub struct SpatialIndex {
    map: BTreeMap<(usize, usize), Vec<NodeId>>,
    reverse: HashMap<NodeId, (usize, usize)>,
}

impl SpatialIndex {
    pub fn new() -> Self {
        Self {
            map: BTreeMap::new(),
            reverse: HashMap::new(),
        }
    }

    pub fn insert(&mut self, start: usize, end: usize, node_id: NodeId) {
        self.map.entry((start, end)).or_default().push(node_id);
        self.reverse.insert(node_id, (start, end));
    }

    pub fn lookup_nodes_from_range(&self, start: usize, end: usize) -> Vec<&NodeId> {
        let (start, end) = if start <= end {
            (start, end)
        } else {
            (end, start)
        };
        self.map
            .iter()
            .filter(|((s, e), _)| start < *e && *s < end)
            .flat_map(|(_, ids)| ids)
            .collect()
    }

    pub fn remove_by_node(&mut self, node_id: &NodeId) {
        if let Some(range) = self.reverse.remove(node_id) {
            if let Some(ids) = self.map.get_mut(&range) {
                ids.retain(|id| id != node_id);
                if ids.is_empty() {
                    self.map.remove(&range);
                }
            }
        }
    }

    pub fn get_range_from_node(&self, node_id: &NodeId) -> Option<(usize, usize)> {
        self.reverse.get(node_id).copied()
    }
}
