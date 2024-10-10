use fxhash::FxHashMap;
use serde::{Serialize, Deserialize};
use std::mem;
use hashbrown::HashMap;
use std::collections::BTreeMap;
use boomphf::hashmap::BoomHashMap;
use sorted_vector_map::SortedVectorMap;


#[derive(Serialize, Deserialize, Clone)]
pub struct TrieNode {
    pub children: SortedVectorMap<u16, (u32, TrieNode)>
}

impl TrieNode {
    pub fn new(capacity: Option<usize>) -> Self {
        TrieNode {
            children: SortedVectorMap::with_capacity(capacity.unwrap_or(0))
        }
    }

    pub fn merge_recursive(&mut self, other: &TrieNode) {
        for (char, other_child) in &other.children {
            let child = self.children
                .entry(*char)
                .or_insert_with(|| (0, TrieNode::new(Some(other_child.1.children.capacity()))));
            child.0 += other_child.0;
            child.1.merge_recursive(&other_child.1);
        }
    }

    pub fn insert_recursive(&mut self, n_gram: &[u16]) { // changed from &[u32] to &[u16]
        if n_gram.len() == 1 { 
            let child = self.children
            .entry(n_gram[0])
            .or_insert_with(|| (0, TrieNode::new(Some(0)))); //leaf node has 0 children
        child.0 += 1;
        return; 
        }
        let child = self.children
            .entry(n_gram[0])
            .or_insert_with(|| (0, TrieNode::new(Some(2_usize.pow(8))))); //TODO: fine tune?
        child.0 += 1;
        child.1.insert_recursive(&n_gram[1..]);
    }

    pub fn size_in_ram_recursive(&self) -> usize {
        let mut size = mem::size_of::<TrieNode>();
        size += self.children.capacity() * mem::size_of::<(u16, (u32, TrieNode))>(); // changed from u32 to u16
        for child in self.children.values() {
            size += child.1.size_in_ram_recursive();
        }
        size
    }

    pub fn shrink_to_fit(&mut self) {
        self.children.shrink_to_fit();
        for child in self.children.values_mut() {
            child.1.shrink_to_fit();
        }
    }

    pub fn find_all_nodes(&self, rule: &[Option<u16>]) -> Vec<(u32, &TrieNode)> { // changed from &[Option<u32>] to &[Option<u16>]
        if rule.len() == 1 { return vec![((self.children.get(&rule[0].unwrap()).unwrap()).0, &self.children.get(&rule[0].unwrap()).unwrap().1)]; }
        else {
            let mut nodes = Vec::<(u32, &TrieNode)>::new();
            match rule[0] {
                None => {
                    for child_node in self.children.values() {
                        nodes.extend(child_node.1.find_all_nodes(&rule[1..]));
                    }
                },
                Some(token) => {
                    if let Some(child_node) = self.children.get(&token) {
                        nodes.extend(child_node.1.find_all_nodes(&rule[1..]));
                    }
                }
            }
            nodes
        }
    }
    
    pub fn get_count(&self, rule: &[Option<u16>]) -> u32 { // changed from &[Option<u32>] to &[Option<u16>]
        if rule.len() == 1 { return self.children.get(&rule[0].unwrap()).unwrap().0; }
        else {
            match rule[0] {
                None => self.children.values()
                    .map(|child| child.1.get_count(&rule[1..]))
                    .sum(),
                Some(token) => self.children.get(&token)
                    .map_or(0, |child| child.1.get_count(&rule[1..]))
            }
        }
    }
}
