use super::{Cpg, Node, NodeId};
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

impl Cpg {
    pub fn get_node_by_offsets(&self, start_byte: usize, end_byte: usize) -> Vec<&Node> {
        let overlapping_ids = self
            .spatial_index
            .lookup_nodes_from_range(start_byte, end_byte);

        overlapping_ids
            .into_iter()
            .filter_map(|id| self.nodes.get(*id))
            .collect()
    }

    pub fn get_node_ids_by_offsets(&self, start_byte: usize, end_byte: usize) -> Vec<NodeId> {
        self.spatial_index
            .lookup_nodes_from_range(start_byte, end_byte)
            .into_iter()
            .cloned()
            .collect()
    }

    pub fn get_node_offsets_by_id(&self, id: &NodeId) -> Option<(usize, usize)> {
        self.spatial_index.get_range_from_node(id)
    }

    pub fn get_smallest_node_id_containing_range(
        &self,
        start_byte: usize,
        end_byte: usize,
    ) -> Option<NodeId> {
        let overlapping_ids = self
            .spatial_index
            .lookup_nodes_from_range(start_byte, end_byte);
        overlapping_ids
            .into_iter()
            .filter(|id| {
                // Only consider nodes that fully contain the range
                let range = self
                    .spatial_index
                    .get_range_from_node(id)
                    .expect("NodeId should have a range");
                range.0 <= start_byte && range.1 >= end_byte
            })
            .min_by_key(|id| {
                let range = self
                    .spatial_index
                    .get_range_from_node(id)
                    .expect("NodeId should have a range");
                range.1 - range.0
            })
            .cloned()
    }
}

#[cfg(test)]
mod tests {

    use crate::cpg::DescendantTraversal;
    use crate::cpg::tests::{create_test_cpg, create_test_node};
    use crate::{cpg::NodeType, desc_trav};

    #[test]
    fn test_spatial_index_basic() {
        let mut cpg = create_test_cpg();
        let node_id1 = cpg.add_node(
            create_test_node(NodeType::Function {
                name_traversal: desc_trav![],
                name: Some("main".to_string()),
            }),
            0,
            10,
        );
        let node_id2 = cpg.add_node(create_test_node(NodeType::Identifier), 5, 15);
        let _node_id3 = cpg.add_node(create_test_node(NodeType::Identifier), 12, 20);

        let overlapping = cpg.spatial_index.lookup_nodes_from_range(8, 12);
        assert_eq!(overlapping.len(), 2);
        assert!(overlapping.contains(&&node_id1));
        assert!(overlapping.contains(&&node_id2));

        let non_overlapping = cpg.spatial_index.lookup_nodes_from_range(21, 25);
        assert!(non_overlapping.is_empty());

        cpg.spatial_index.remove_by_node(&node_id2);
        let after_removal = cpg.spatial_index.lookup_nodes_from_range(8, 12);
        assert_eq!(after_removal.len(), 1);
        assert!(after_removal.contains(&&node_id1));
    }

    #[test]
    fn test_spatial_index_edge_cases() {
        let mut cpg = create_test_cpg();

        // Test zero-width ranges
        let _node_id = cpg.add_node(create_test_node(NodeType::Identifier), 5, 5);
        let overlapping = cpg.spatial_index.lookup_nodes_from_range(5, 5);
        assert!(overlapping.is_empty()); // Zero-width ranges don't overlap

        // Test exact boundaries
        let node_id2 = cpg.add_node(
            create_test_node(NodeType::Function {
                name_traversal: desc_trav![],
                name: Some("main".to_string()),
            }),
            0,
            10,
        );
        let exact_match = cpg.spatial_index.lookup_nodes_from_range(0, 10);
        // Note: The spatial index includes the first node added (root), so count should be 2
        assert_eq!(exact_match.len(), 2);
        assert!(exact_match.contains(&&node_id2));

        // Test adjacent ranges
        let _node_id3 = cpg.add_node(create_test_node(NodeType::Statement), 10, 20);
        let adjacent = cpg.spatial_index.lookup_nodes_from_range(10, 10);
        assert!(adjacent.is_empty()); // Adjacent ranges shouldn't overlap
    }

    #[test]
    fn test_get_smallest_node_containing_range() {
        let mut cpg = create_test_cpg();
        let large_node = cpg.add_node(
            create_test_node(NodeType::Function {
                name_traversal: desc_trav![],
                name: Some("main".to_string()),
            }),
            0,
            100,
        );
        let medium_node = cpg.add_node(create_test_node(NodeType::Block), 10, 50);
        let small_node = cpg.add_node(create_test_node(NodeType::Identifier), 20, 30);

        let result = cpg.get_smallest_node_id_containing_range(25, 26);
        assert_eq!(result, Some(small_node));

        let result2 = cpg.get_smallest_node_id_containing_range(15, 45);
        assert_eq!(result2, Some(medium_node));

        let result3 = cpg.get_smallest_node_id_containing_range(5, 95);
        assert_eq!(result3, Some(large_node));

        let result4 = cpg.get_smallest_node_id_containing_range(200, 300);
        assert_eq!(result4, None);
    }

    #[test]
    fn test_get_node_by_offsets() {
        let mut cpg = create_test_cpg();
        let node1 = cpg.add_node(
            create_test_node(NodeType::Function {
                name_traversal: desc_trav![],
                name: Some("main".to_string()),
            }),
            0,
            10,
        );
        let node2 = cpg.add_node(create_test_node(NodeType::Identifier), 5, 15);

        let nodes = cpg.get_node_by_offsets(8, 12);
        assert_eq!(nodes.len(), 2);

        let node_ids = cpg.get_node_ids_by_offsets(8, 12);
        assert_eq!(node_ids.len(), 2);
        assert!(node_ids.contains(&node1));
        assert!(node_ids.contains(&node2));
    }
}
