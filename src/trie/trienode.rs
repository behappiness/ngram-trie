use serde::{Serialize, Deserialize};
use sorted_vector_map::SortedVectorMap;
use rclite::Arc;
use rayon::prelude::*;

#[derive(Serialize, Deserialize, Debug)]
pub struct TrieNode {
    pub children: SortedVectorMap<u16, Arc<TrieNode>>,
    pub count: u32,
}

impl TrieNode {
    pub fn new(capacity: Option<usize>) -> Self {
        TrieNode {
            children: SortedVectorMap::with_capacity(capacity.unwrap_or(0)),
            count: 0,
        }
    }

    pub fn merge(&mut self, other: Arc<TrieNode>) {
        self.count += other.count;
        for (key, other_child) in other.children.iter() {
            if let Some(child) = self.children.get_mut(key) {
                Arc::get_mut(child).unwrap().merge(other_child.clone());
            } else {
                self.children.insert(*key, other_child.clone());
            }
        }
    }

    pub fn insert(&mut self, n_gram: &[u16]) {
        self.count += 1;
        if n_gram.is_empty() {
            return;
        }

        if let Some(child) = self.children.get_mut(&n_gram[0]) {
            Arc::get_mut(child).unwrap().insert(&n_gram[1..]);
        } else {
            let new_child = Arc::new(TrieNode::new(None));
            self.children.insert(n_gram[0], new_child);
            Arc::get_mut(&mut self.children.get_mut(&n_gram[0]).unwrap()).unwrap().insert(&n_gram[1..]);
        }
    }

    pub fn shrink_to_fit(&mut self) {
        self.children.shrink_to_fit();

        for child in self.children.values_mut() {
            Arc::get_mut(child).unwrap().shrink_to_fit();
        }
    }

    pub fn find_all_nodes(&self, rule: &[Option<u16>]) -> Vec<Arc<TrieNode>> {
        let result = match rule.len() {
            0 => return vec![],
            1 => {
                match rule[0] {
                    None => {
                        self.children.values().cloned().collect()
                    },
                    Some(token) => {
                        if let Some(child) = self.children.get(&token).cloned() {
                            vec![child]
                        } else {
                            vec![]
                        }
                    }
                }
            },
            _ => {
                match rule[0] {
                    None => {
                        self.children.par_iter()
                            .flat_map(|(_, child)| child.find_all_nodes(&rule[1..]))
                            .collect()
                    },
                    Some(token) => {
                        if let Some(child) = self.children.get(&token) {
                            child.find_all_nodes(&rule[1..])
                        } else {
                            vec![]
                        }
                    }
                }
            }
        };
        result
    }
    
    pub fn get_count(&self, rule: &[Option<u16>]) -> u32 {
        let count = if rule.len() == 0 { self.count } else {
            match rule[0] {
                None => self.children.par_iter()
                    .map(|(_, child)| child.get_count(&rule[1..]))
                    .sum(),
                Some(token) => self.children.get(&token)
                    .map_or(0, |child| child.get_count(&rule[1..]))
            }
        };
        count
    }

    pub fn count_ns(&self) -> (u32, u32, u32, u32, u32, u32) {
        let mut n1 = 0;
        let mut n2 = 0;
        let mut n3 = 0;
        let mut n4 = 0;
        let mut nodes = 1;
        let mut rest = 0;
        match self.count {
            1 => n1 += 1,
            2 => n2 += 1,
            3 => n3 += 1,
            4 => n4 += 1,
            _ => rest += 1
        }
        for child in self.children.values() {
            let (c1, c2, c3, c4, _nodes, _rest) = child.count_ns();
            n1 += c1;
            n2 += c2;
            n3 += c3;
            n4 += c4;
            nodes += _nodes;
            rest += _rest;
        }
        (n1, n2, n3, n4, nodes, rest)
    }

}