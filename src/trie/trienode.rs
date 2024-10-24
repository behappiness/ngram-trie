use fxhash::FxHashMap;
use serde::{Serialize, Deserialize};
use std::mem;
use hashbrown::HashMap;
use std::collections::BTreeMap;
use boomphf::hashmap::BoomHashMap;
use sorted_vector_map::SortedVectorMap;
use rayon::prelude::*;


#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TrieNode {
    pub children: SortedVectorMap<u16, Box<TrieNode>>, // changed from u32 to u16
    pub count: u32
}

impl TrieNode {
    pub fn new(capacity: Option<usize>) -> Self {
        TrieNode {
            children: SortedVectorMap::with_capacity(capacity.unwrap_or(0)),
            count: 0,
        }
    }

    pub fn merge(&mut self, other: &TrieNode) {
        self.count += other.count;
        for (char, other_child) in &other.children {
            self.children
                .entry(*char)
                .or_insert_with(|| Box::new(TrieNode::new(Some(other.children.len()))))
                .merge(other_child);
        }
    }

    pub fn insert(&mut self, n_gram: &[u16]) { // changed from &[u32] to &[u16]
        self.count += 1;
        match n_gram.len() {
            0 => return,
            1 => {
                self.children
                .entry(n_gram[0])
                .or_insert_with(|| Box::new(TrieNode::new(None)))
                .insert(&n_gram[1..]);
            },
            _ => {
                self.children
                .entry(n_gram[0])
                .or_insert_with(|| Box::new(TrieNode::new(None)))//2^5 is the default capacity of a TrieNode
                .insert(&n_gram[1..]);
            }
        }
    }

    pub fn size_in_ram(&self) -> usize {
        let mut size = mem::size_of::<TrieNode>();
        size += self.children.capacity() * mem::size_of::<(u16, Box<TrieNode>)>(); // changed from u32 to u16
        for child in self.children.values() {
            size += child.size_in_ram();
        }
        size
    }

    /// Shrinks the children vector to fit the number of elements. Starting from the leaf nodes.
    pub fn shrink_to_fit(&mut self) {
        for child in self.children.values_mut() {
            child.shrink_to_fit();
        }
        self.children.shrink_to_fit();
    }

    pub fn find_all_nodes(&self, rule: &[Option<u16>]) -> Vec<&TrieNode> { // changed from &[Option<u32>] to &[Option<u16>]
        if rule.is_empty() { return vec![self]; }
        else {
            let mut nodes = Vec::<&TrieNode>::new();
            match rule[0] {
                None => {
                    for child_node in self.children.values() {
                        nodes.extend(child_node.find_all_nodes(&rule[1..]));
                    }
                },
                Some(token) => {
                    if let Some(child_node) = self.children.get(&token) {
                        nodes.extend(child_node.find_all_nodes(&rule[1..]));
                    }
                }
            }
            nodes
        }
    }
    
    pub fn get_count(&self, rule: &[Option<u16>]) -> u32 { // changed from &[Option<u32>] to &[Option<u16>]
        if rule.len() == 0 { return self.count; }
        else {
            match rule[0] {
                None => self.children.values()
                    .map(|child| child.get_count(&rule[1..]))
                    .sum(),
                Some(token) => self.children.get(&token)
                    .map_or(0, |child| child.get_count(&rule[1..]))
            }
        }
    }

    pub fn count_ns(&self) -> (u32, u32, u32, u32) {
        let mut n1 = 0;
        let mut n2 = 0;
        let mut n3 = 0;
        let mut n4 = 0;
        match self.count {
            1 => n1 += 1,
            2 => n2 += 1,
            3 => n3 += 1,
            4 => n4 += 1,
            _ => ()
        }
        for child in self.children.values() {
            let (c1, c2, c3, c4) = child.count_ns();
            n1 += c1;
            n2 += c2;
            n3 += c3;
            n4 += c4;
        }
        (n1, n2, n3, n4)
    }

    pub fn semi_deep_clone(&self) -> TrieNode {
        let mut cloned_node = TrieNode {
            count: self.count,
            children: SortedVectorMap::with_capacity(self.children.capacity()),
        };
        for (key, child) in &self.children {
            cloned_node.children.insert(*key, Box::new( TrieNode {
                children: SortedVectorMap::with_capacity(0),
                count: child.count,
            }));
        }
        cloned_node
    }
}
