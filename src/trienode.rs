use fxhash::FxHashMap;
use serde::{Serialize, Deserialize};
use std::mem;

#[derive(Serialize, Deserialize, Clone)]
pub struct TrieNode {
    pub children: FxHashMap<u16, Box<TrieNode>>, // changed from u32 to u16
    pub count: u32
}

impl TrieNode {
    pub fn new() -> Self {
        TrieNode {
            children: FxHashMap::default(),
            count: 0,
        }
    }

    pub fn merge_recursive(&mut self, other: &TrieNode) {
        self.count += other.count;
        for (char, other_child) in &other.children {
            self.children
                .entry(*char)
                .or_insert_with(|| Box::new(TrieNode::new()))
                .merge_recursive(other_child);
        }
    }

    pub fn insert_recursive(&mut self, n_gram: &[u16]) { // changed from &[u32] to &[u16]
        self.count += 1;
        if n_gram.len() == 0 { return; }
        self.children
            .entry(n_gram[0])
            .or_insert_with(|| Box::new(TrieNode::new()))
            .insert_recursive(&n_gram[1..]);
    }

    pub fn size_in_ram_recursive(&self) -> usize {
        let mut size = mem::size_of::<TrieNode>();
        size += self.children.capacity() * mem::size_of::<(u16, Box<TrieNode>)>(); // changed from u32 to u16
        for child in self.children.values() {
            size += child.size_in_ram_recursive();
        }
        size
    }

    pub fn find_all_nodes(&self, rule: &[Option<u16>]) -> Vec<&TrieNode> { // changed from &[Option<u32>] to &[Option<u16>]
        if rule.len() == 0 { return vec![self]; }
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
}
