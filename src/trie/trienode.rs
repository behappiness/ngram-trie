use fxhash::FxHashMap;
use serde::{Serialize, Deserialize};
use std::mem;
use hashbrown::HashMap;
use std::collections::BTreeMap;
use boomphf::hashmap::BoomHashMap;
use sorted_vector_map::SortedVectorMap;
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering;

#[derive(Serialize, Deserialize, Debug)]
pub struct TrieNode {
    pub children: RwLock<SortedVectorMap<u16, Arc<TrieNode>>>, // changed from u32 to u16
    pub count: AtomicU32
}

impl TrieNode {
    pub fn new(capacity: Option<usize>) -> Self {
        TrieNode {
            children: RwLock::new(SortedVectorMap::with_capacity(capacity.unwrap_or(0))),
            count: AtomicU32::new(0),
        }
    }

    pub fn merge(&self, other: Arc<TrieNode>) {
        self.count.fetch_add(other.count.load(Ordering::SeqCst), Ordering::SeqCst);
        for (key, other_child) in other.children.read().unwrap().iter() {
            if let Some(child) = self.children.read().unwrap().get(key) {
                child.merge(other_child.clone());
            } else {
                self.children.write().unwrap().insert(*key, other_child.clone());
            }
        }
    }

    pub fn insert(&self, n_gram: &[u16]) {
        self.count.fetch_add(1, Ordering::SeqCst);
        if n_gram.is_empty() {
            return;
        }

        let first = n_gram[0];
        let mut children = self.children.write().unwrap();

        if let Some(child) = children.get(&first).cloned() {
            // Release the write lock before recursing
            drop(children);
            child.insert(&n_gram[1..]);
        } else {
            // Create a new child node
            let new_child = Arc::new(TrieNode::new(None));
            children.insert(first, new_child.clone());
            // Release the write lock before recursing
            drop(children);
            new_child.insert(&n_gram[1..]);
        }
    }

    /// Shrinks the children vector to fit the number of elements. Starting from the leaf nodes.
    pub fn shrink_to_fit(&self) {
        // First, collect all child nodes while holding a read lock
        let children: Vec<Arc<TrieNode>> = {
            let read_guard = self.children.read().unwrap();
            read_guard.values().cloned().collect()
        };

        // Now, shrink the children vector
        self.children.write().unwrap().shrink_to_fit();

        // Recursively call shrink_to_fit on each child node
        for child in children {
            child.shrink_to_fit();
        }
    }

    pub fn find_all_nodes(&self, rule: &[Option<u16>]) -> Vec<Arc<TrieNode>> {
        match rule.len() {
            0 => return vec![],
            1 => {
                match rule[0] {
                    None => {
                        // Clone the children while holding the read lock
                        let children: Vec<Arc<TrieNode>> = self.children.read().unwrap().values().cloned().collect();
                        return children;
                    },
                    Some(token) => {
                        // Clone the child while holding the read lock
                        if let Some(child) = self.children.read().unwrap().get(&token).cloned() {
                            return vec![child];
                        } else {
                            return vec![];
                        }
                    }
                }
            },
            _ => {
                match rule[0] {
                    None => {
                        // Clone the children while holding the read lock
                        let children: Vec<Arc<TrieNode>> = self.children.read().unwrap().values().cloned().collect();
                        return children.into_iter().flat_map(|child| child.find_all_nodes(&rule[1..])).collect();
                    },
                    Some(token) => {
                        // Clone the child while holding the read lock
                        if let Some(child) = self.children.read().unwrap().get(&token).cloned() {
                            return child.find_all_nodes(&rule[1..]);
                        } else {
                            return vec![];
                        }
                    }
                }
            }
        }
    }
    
    pub fn get_count(&self, rule: &[Option<u16>]) -> u32 { // changed from &[Option<u32>] to &[Option<u16>]
        if rule.len() == 0 { return self.count.load(Ordering::SeqCst); }
        else {
            match rule[0] {
                None => self.children.read().unwrap().values()
                    .map(|child| child.get_count(&rule[1..]))
                    .sum(),
                Some(token) => self.children.read().unwrap().get(&token)
                    .map_or(0, |child| child.get_count(&rule[1..]))
            }
        }
    }

    pub fn count_ns(&self) -> (u32, u32, u32, u32, u32) {
        let mut n1 = 0;
        let mut n2 = 0;
        let mut n3 = 0;
        let mut n4 = 0;
        let mut nodes = 1;
        match self.count.load(Ordering::SeqCst) {
            1 => n1 += 1,
            2 => n2 += 1,
            3 => n3 += 1,
            4 => n4 += 1,
            _ => ()
        }
        for child in self.children.read().unwrap().values() {
            let (c1, c2, c3, c4, _nodes) = child.count_ns();
            n1 += c1;
            n2 += c2;
            n3 += c3;
            n4 += c4;
            nodes += _nodes;
        }
        (n1, n2, n3, n4, nodes)
    }

    // pub fn semi_deep_clone(&self) -> TrieNode {
    //     let mut cloned_node = TrieNode {
    //         count: self.count,
    //         children: SortedVectorMap::with_capacity(self.children.capacity()),
    //     };
    //     for (key, child) in &self.children {
    //         cloned_node.children.insert(*key, Box::new( TrieNode {
    //             children: SortedVectorMap::with_capacity(0),
    //             count: child.count,
    //         }));
    //     }
    //     cloned_node
    // }
}
