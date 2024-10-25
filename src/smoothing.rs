use crate::trie::NGramTrie;
use simple_tqdm::ParTqdm;
use std::time::Instant;
use serde::{Serialize, Deserialize};
use rclite::Arc;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};
use quick_cache::sync::Cache;
use lazy_static::lazy_static;
use sorted_vector_map::SortedVectorSet;

// the dataset size matters as well
const CACHE_SIZE_S_C: usize = 233*16104*32; //(rules+25%)*keys = RULES*KEYS
const CACHE_SIZE_S_N: usize = 233*3*32; //(rules+25%) = RULES*1.25

lazy_static! {
    pub static ref CACHE_S_C: Cache<Vec<Option<u16>>, f64> = Cache::new(CACHE_SIZE_S_C);
    pub static ref CACHE_S_N: Cache<Vec<Option<u16>>, (u32, u32, u32)> = Cache::new(CACHE_SIZE_S_N);
}   

pub trait Smoothing: Sync + Send {
    fn smoothing(&self, trie: Arc<NGramTrie>, rule: &[Option<u16>]) -> f64;
    fn save(&self, filename: &str);
    fn load(&mut self, filename: &str);
    fn reset_cache(&self);
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ModifiedBackoffKneserNey {
    pub d1: f64,
    pub d2: f64,
    pub d3: f64,
    pub uniform: f64
}

impl ModifiedBackoffKneserNey {
    pub fn new(trie: Arc<NGramTrie>) -> Self {
        let (_d1, _d2, _d3, _uniform) = Self::calculate_d_values(trie);
        ModifiedBackoffKneserNey {
            d1: _d1,
            d2: _d2,
            d3: _d3,
            uniform: _uniform
        }
    }

    pub fn calculate_d_values(trie: Arc<NGramTrie>) -> (f64, f64, f64, f64) {
        println!("----- Calculating d values -----");
        let start = Instant::now();
        let n1 = Arc::new(AtomicU32::new(0));
        let n2 = Arc::new(AtomicU32::new(0));
        let n3 = Arc::new(AtomicU32::new(0));
        let n4 = Arc::new(AtomicU32::new(0));
        let nodes = Arc::new(AtomicU32::new(0));
        trie.root.children.par_iter().tqdm().for_each(|(_, child)| {
            let (c1, c2, c3, c4, _nodes, _rest) = child.count_ns();
            n1.fetch_add(c1, Ordering::SeqCst);
            n2.fetch_add(c2, Ordering::SeqCst);
            n3.fetch_add(c3, Ordering::SeqCst);
            n4.fetch_add(c4, Ordering::SeqCst);
            nodes.fetch_add(_nodes, Ordering::SeqCst);
        });

        let n1 = n1.load(Ordering::SeqCst);
        let n2 = n2.load(Ordering::SeqCst);
        let n3 = n3.load(Ordering::SeqCst);
        let n4 = n4.load(Ordering::SeqCst);
        let nodes = nodes.load(Ordering::SeqCst);

        println!("Nodes: {}", nodes);
        let uniform = 1.0 / trie.root.children.len() as f64;

        if n1 == 0 || n2 == 0 || n3 == 0 || n4 == 0 {
            return (0.1, 0.2, 0.3, uniform);
        }

        let y = n1 as f64 / (n1 + 2 * n2) as f64;
        let d1 = 1.0 - 2.0 * y * (n2 as f64 / n1 as f64);
        let d2 = 2.0 - 3.0 * y * (n3 as f64 / n2 as f64);
        let d3 = 3.0 - 4.0 * y * (n4 as f64 / n3 as f64);
        let elapsed = start.elapsed();
        println!("Time taken: {:.2?}", elapsed);
        println!("Smoothing calculated, d1: {}, d2: {}, d3: {}, uniform: {}", d1, d2, d3, uniform);
        (d1, d2, d3, uniform)
    }

    pub fn count_unique_ns(trie: Arc<NGramTrie>, rule: Vec<Option<u16>>) -> (u32, u32, u32) {
        if let Some(cached_value) = CACHE_S_N.get(&rule) {
            return cached_value;
        }
        let mut n1 = SortedVectorSet::<u16>::new();
        let mut n2 = SortedVectorSet::<u16>::new();
        let mut n3 = SortedVectorSet::<u16>::new();
        for node in trie.find_all_nodes(rule.clone()).iter() {
            for (key, child) in node.children.iter() {
                match child.count {
                    1 => { n1.insert(*key); },
                    2 => { n2.insert(*key); },
                    _ => { n3.insert(*key); }
                }
            }
        }
        let result = (n1.len() as u32, n2.len() as u32, n3.len() as u32);
        CACHE_S_N.insert(rule, result);
        result
    }

    pub fn init_cache(&self) {
        CACHE_S_C.insert(vec![], self.uniform);
    }
}

//From Chen & Goodman 1998
impl Smoothing for ModifiedBackoffKneserNey {
    fn save(&self, filename: &str) {
        let _file = filename.to_owned() + ".smoothing";
        let serialized = bincode::serialize(self).unwrap();
        std::fs::write(_file, serialized).unwrap();
    }

    fn load(&mut self, filename: &str) {
        let _file = filename.to_owned() + ".smoothing";
        let serialized = std::fs::read(_file).unwrap();
        *self = bincode::deserialize(&serialized).unwrap();
    }

    fn reset_cache(&self) {
        CACHE_S_C.clear();
        CACHE_S_N.clear();
        self.init_cache();
    }

    fn smoothing(&self, trie: Arc<NGramTrie>, rule: &[Option<u16>]) -> f64 {
        if let Some(cached_value) = CACHE_S_C.get(rule) {
            return cached_value;
        }

        let recursive = self.smoothing(trie.clone(), &rule[1..]);

        //let w_i = &rule[rule.len() - 1];
        let w_i_minus_1 = &rule[..rule.len() - 1];

        let (n1, n2, n3) = Self::count_unique_ns(trie.clone(), w_i_minus_1.to_vec());

        let c_i_minus_1 = trie.get_count(&w_i_minus_1);
        let c_i = trie.get_count(&rule);

        let d = match c_i {
            0 => 0.0,
            1 => self.d1,
            2 => self.d2,
            _ => self.d3
        };

        let gamma = (self.d1 * n1 as f64 + self.d2 * n2 as f64 + self.d3 * n3 as f64) / c_i_minus_1 as f64;

        let result = (c_i as f64 - d).max(0.0) / c_i_minus_1 as f64 + gamma * recursive;
        CACHE_S_C.insert(rule.to_vec(), result);
        result
    }
}