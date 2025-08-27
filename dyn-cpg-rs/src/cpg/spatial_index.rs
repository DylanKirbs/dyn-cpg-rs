use super::{Cpg, Node, NodeId};
use std::collections::{BTreeMap, HashMap};

#[allow(dead_code)] // We have a few methods that need not be used, but are useful to have and may become useful in the incremental side of things
pub trait SpatialIndex
where
    Self: Default,
    Self: Sized,
    Self: Clone,
    Self: std::fmt::Debug,
    Self::NodeId: std::cmp::Eq + std::hash::Hash + Copy,
{
    type NodeId;

    /// Insert a node's span into the index.
    fn insert(&mut self, id: Self::NodeId, start: usize, end: usize);

    /// Delete a node's span from the index.
    fn delete(&mut self, id: Self::NodeId);

    /// Edit a nodes span within the index.
    fn edit(&mut self, id: Self::NodeId, start: usize, end: usize);

    /// Query all nodes covering a point.
    fn get_nodes_covering_point(&self, pos: usize) -> Vec<Self::NodeId>;

    /// Query all nodes overlapping a range.
    fn get_nodes_covering_range(&self, start: usize, end: usize) -> Vec<Self::NodeId>;

    /// Query all nodes contained within a range.
    fn get_nodes_within_range(&self, start: usize, end: usize) -> Vec<Self::NodeId>;

    /// Query the span of a given node.
    fn get_node_span(&self, id: Self::NodeId) -> Option<(usize, usize)>;
}

#[derive(Debug, Clone, Default)]
pub struct BTreeIndex {
    map: BTreeMap<(usize, usize), Vec<NodeId>>,
    reverse: HashMap<NodeId, (usize, usize)>,
}

impl SpatialIndex for BTreeIndex {
    type NodeId = NodeId;

    fn insert(&mut self, id: Self::NodeId, start: usize, end: usize) {
        self.map.entry((start, end)).or_default().push(id);
        self.reverse.insert(id, (start, end));
    }

    fn delete(&mut self, id: Self::NodeId) {
        if let Some((start, end)) = self.reverse.remove(&id) {
            let key = (start, end);
            let mut remove_key = false;
            if let Some(ids) = self.map.get_mut(&key) {
                ids.retain(|i| i != &id);
                if ids.is_empty() {
                    remove_key = true;
                }
            }
            if remove_key {
                self.map.remove(&key);
            }
        }
    }

    fn edit(&mut self, id: Self::NodeId, start: usize, end: usize) {
        if let Some((old_start, old_end)) = self.reverse.remove(&id) {
            let key = (old_start, old_end);
            let mut remove_key = false;
            if let Some(ids) = self.map.get_mut(&key) {
                ids.retain(|i| i != &id);
                if ids.is_empty() {
                    remove_key = true;
                }
            }
            if remove_key {
                self.map.remove(&key);
            }
        }
        self.map.entry((start, end)).or_default().push(id);
        self.reverse.insert(id, (start, end));
    }

    fn get_nodes_covering_point(&self, pos: usize) -> Vec<Self::NodeId> {
        self.map
            .iter()
            .filter(|((s, e), _)| *s <= pos && pos < *e)
            .flat_map(|(_, ids)| ids)
            .cloned()
            .collect()
    }

    fn get_nodes_covering_range(&self, start: usize, end: usize) -> Vec<Self::NodeId> {
        let (start, end) = if start <= end {
            (start, end)
        } else {
            (end, start)
        };
        self.map
            .iter()
            .filter(|((s, e), _)| start <= *e && *s < end)
            .flat_map(|(_, ids)| ids)
            .cloned()
            .collect()
    }

    fn get_nodes_within_range(&self, start: usize, end: usize) -> Vec<Self::NodeId> {
        let (start, end) = if start <= end {
            (start, end)
        } else {
            (end, start)
        };
        self.map
            .iter()
            .filter(|((s, e), _)| start < *e && *s < end)
            .flat_map(|(_, ids)| ids)
            .cloned()
            .collect()
    }

    fn get_node_span(&self, id: Self::NodeId) -> Option<(usize, usize)> {
        self.reverse.get(&id).cloned()
    }
}

impl Cpg {
    pub fn get_node_by_offsets(&self, start_byte: usize, end_byte: usize) -> Vec<&Node> {
        let overlapping_ids = self
            .spatial_index
            .get_nodes_covering_range(start_byte, end_byte);

        overlapping_ids
            .into_iter()
            .filter_map(|id| self.nodes.get(id))
            .collect()
    }

    pub fn get_node_ids_by_offsets(&self, start_byte: usize, end_byte: usize) -> Vec<NodeId> {
        self.spatial_index
            .get_nodes_covering_range(start_byte, end_byte)
            .into_iter()
            .collect::<Vec<_>>()
    }

    pub fn get_node_offsets_by_id(&self, id: &NodeId) -> Option<(usize, usize)> {
        self.spatial_index.get_node_span(*id)
    }

    pub fn get_smallest_node_id_containing_range(
        &self,
        start_byte: usize,
        end_byte: usize,
    ) -> Option<NodeId> {
        let overlapping_ids = self
            .spatial_index
            .get_nodes_covering_range(start_byte, end_byte);
        overlapping_ids
            .into_iter()
            .filter(|id| {
                // Only consider nodes that fully contain the range
                let range = self
                    .spatial_index
                    .get_node_span(*id)
                    .expect("NodeId should have a range");
                range.0 <= start_byte && range.1 >= end_byte
            })
            .min_by_key(|id| {
                let range = self
                    .spatial_index
                    .get_node_span(*id)
                    .expect("NodeId should have a range");
                range.1 - range.0
            })
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_spatial_index_incremental_update_simulation() {
        // Simulate a pipeline: add nodes, "edit" (remove and add), and check index consistency
        let mut cpg = create_test_cpg();
        // Add initial nodes
        let n1 = cpg.add_node(
            create_test_node(NodeType::Function {
                name_traversal: desc_trav![],
                name: Some("main".to_string()),
            }),
            0,
            100,
        );
        let n2 = cpg.add_node(create_test_node(NodeType::Block), 10, 50);
        let n3 = cpg.add_node(create_test_node(NodeType::Identifier), 20, 30);

        // Simulate an edit: remove n2, add a new node in its place
        cpg.spatial_index.delete(n2);
        let n2b = cpg.add_node(create_test_node(NodeType::Block), 12, 48);

        // Simulate another edit: remove n3, add a new node
        cpg.spatial_index.delete(n3);
        let n3b = cpg.add_node(create_test_node(NodeType::Identifier), 22, 28);

        // Now check that get_smallest_node_id_containing_range returns the correct node
        let result = cpg.get_smallest_node_id_containing_range(25, 26);
        assert_eq!(result, Some(n3b));
        let result2 = cpg.get_smallest_node_id_containing_range(15, 45);
        assert_eq!(result2, Some(n2b));
        let result3 = cpg.get_smallest_node_id_containing_range(5, 95);
        assert_eq!(result3, Some(n1));

        // Check that deleted nodes are not returned
        let deleted_result = cpg.get_smallest_node_id_containing_range(11, 49);
        assert_ne!(deleted_result, Some(n2));
        let deleted_result2 = cpg.get_smallest_node_id_containing_range(23, 27);
        assert_ne!(deleted_result2, Some(n3));

        // Check that all node IDs returned by the spatial index exist in the CPG
        for id in cpg.spatial_index.get_nodes_covering_range(0, 100) {
            assert!(
                cpg.get_node_by_id(&id).is_some(),
                "NodeId {:?} in spatial index but not in CPG",
                id
            );
        }
    }

    use crate::cpg::DescendantTraversal;
    use crate::cpg::spatial_index::SpatialIndex;
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

        let overlapping = cpg.spatial_index.get_nodes_covering_range(8, 12);
        assert_eq!(overlapping.len(), 2);
        assert!(overlapping.contains(&&node_id1));
        assert!(overlapping.contains(&&node_id2));

        let non_overlapping = cpg.spatial_index.get_nodes_covering_range(21, 25);
        assert!(non_overlapping.is_empty());

        cpg.spatial_index.delete(node_id2);
        let after_removal = cpg.spatial_index.get_nodes_covering_range(8, 12);
        assert_eq!(after_removal.len(), 1);
        assert!(after_removal.contains(&&node_id1));
    }

    #[test]
    fn test_spatial_index_edge_cases() {
        let mut cpg = create_test_cpg();

        // Test zero-width ranges
        let _node_id = cpg.add_node(create_test_node(NodeType::Identifier), 5, 5);
        let overlapping = cpg.spatial_index.get_nodes_covering_range(5, 5);
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
        let exact_match = cpg.spatial_index.get_nodes_covering_range(0, 10);
        // Note: The spatial index includes the first node added (root), so count should be 2
        assert_eq!(exact_match.len(), 2);
        assert!(exact_match.contains(&&node_id2));

        // Test adjacent ranges
        let _node_id3 = cpg.add_node(create_test_node(NodeType::Statement), 10, 20);
        let adjacent = cpg.spatial_index.get_nodes_covering_range(10, 10);
        assert!(!adjacent.is_empty());
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
