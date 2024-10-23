use crate::trie::NGramTrie;
use sorted_vector_map::SortedVectorSet;
use simple_tqdm::ParTqdm;
use std::time::Instant;
use serde::{Serialize, Deserialize};
use std::sync::{Arc, Mutex, RwLock};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};
use cached::proc_macro::cached;
use quick_cache::sync::Cache;
use lazy_static::lazy_static;
use lru::LruCache;
use std::num::NonZero;
use hashbrown::HashSet;

pub const BATCH_SIZE: usize = 15_000_000;
pub const CACHE_SIZE: usize = 10_000_000;

lazy_static! {
    static ref CACHE: RwLock<LruCache<Vec<Option<u16>>, f64>> = RwLock::new(LruCache::new(NonZero::new(CACHE_SIZE).unwrap()));
}

pub trait Smoothing: Sync + Send {
    fn smoothing(&self, trie: Arc<NGramTrie>, rule: &[Option<u16>]) -> f64;
    fn save(&self, filename: &str);
    fn load(&mut self, filename: &str);
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ModifiedBackoffKneserNey {
    pub d1: f64,
    pub d2: f64,
    pub d3: f64,
    pub uniform: f64
}

impl ModifiedBackoffKneserNey {
    pub fn new(trie: &NGramTrie) -> Self {
        let (_d1, _d2, _d3, _uniform) = Self::calculate_d_values(trie);
        ModifiedBackoffKneserNey {
            d1: _d1,
            d2: _d2,
            d3: _d3,
            uniform: _uniform
        }
    }

    pub fn calculate_d_values(trie: &NGramTrie) -> (f64, f64, f64, f64) {
        println!("----- Calculating d values -----");
        let start = Instant::now();
        let n1 = Arc::new(AtomicU32::new(0));
        let n2 = Arc::new(AtomicU32::new(0));
        let n3 = Arc::new(AtomicU32::new(0));
        let n4 = Arc::new(AtomicU32::new(0));
        let mut nodes = Vec::new();
        for i in 1..=trie.n_gram_max_length {
            let rule: Vec<Option<u16>> = vec![None; i as usize];
            nodes.extend(trie.find_all_nodes(&rule));
        }

        println!("Number of nodes: {}", nodes.len());

        let batch_size = BATCH_SIZE;
        let num_batches = (nodes.len() as f64 / batch_size as f64).ceil() as usize;

        (0..num_batches).into_par_iter().tqdm().for_each(|batch| {
            let start = batch * batch_size;
            let end = (start + batch_size).min(nodes.len());
            let mut local_n1 = 0;
            let mut local_n2 = 0;
            let mut local_n3 = 0;
            let mut local_n4 = 0;
            for node in &nodes[start..end] {
                match node.count {
                    1 => local_n1 += 1,
                    2 => local_n2 += 1,
                    3 => local_n3 += 1,
                    4 => local_n4 += 1,
                    _ => ()
                }
            }
            n1.fetch_add(local_n1 as u32, Ordering::SeqCst);
            n2.fetch_add(local_n2 as u32, Ordering::SeqCst);
            n3.fetch_add(local_n3 as u32, Ordering::SeqCst);
            n4.fetch_add(local_n4 as u32, Ordering::SeqCst);
        });

        let n1 = n1.load(Ordering::SeqCst);
        let n2 = n2.load(Ordering::SeqCst);
        let n3 = n3.load(Ordering::SeqCst);
        let n4 = n4.load(Ordering::SeqCst);

        let uniform = 1.0 / trie.root.children.len() as f64;

        if n1 == 0 || n2 == 0 || n3 == 0 || n4 == 0 {
            return (0.1, 0.2, 0.3, uniform);
        }

        let y = n1 as f64 / (n1 + 2 * n2) as f64;
        let d1 = 1.0 - 2.0 * y * (n2 as f64 / n1 as f64);
        let d2 = 2.0 - 3.0 * y * (n3 as f64 / n2 as f64);
        let d3 = 3.0 - 4.0 * y * (n4 as f64 / n3 as f64);
        let elapsed = start.elapsed();
        println!("Time taken: {:?}", elapsed);
        println!("Smoothing calculated, d1: {}, d2: {}, d3: {}, uniform: {}", d1, d2, d3, uniform);
        (d1, d2, d3, uniform)
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

    fn smoothing(&self, trie: Arc<NGramTrie>, rule: &[Option<u16>]) -> f64 {
        if let Some(&cached_value) = CACHE.read().unwrap().peek(rule) {
            return cached_value;
        }

        if rule.is_empty() {
            return self.uniform;
        }

        let W_i = &rule[rule.len() - 1];
        let W_i_minus_1 = &rule[..rule.len() - 1];

        let C_i = trie.get_count(&rule);
        let C_i_minus_1 = trie.get_count(&W_i_minus_1);

        let d = match C_i {
            0 => 0.0,
            1 => self.d1,
            2 => self.d2,
            _ => self.d3
        };

        let (n1, n2, n3) = count_unique_ns(trie.clone(), W_i_minus_1.to_vec());

        let gamma = (self.d1 * n1 as f64 + self.d2 * n2 as f64 + self.d3 * n3 as f64) / C_i_minus_1 as f64;

        let result = ((C_i as f64 - d).max(0.0) / C_i_minus_1 as f64 + gamma * self.smoothing(trie, &rule[1..])).into();
        CACHE.write().unwrap().put(rule.to_vec(), result);
        result
    }
}

//#[cached]
pub fn count_unique_ns(trie: Arc<NGramTrie>, rule: Vec<Option<u16>>) -> (u32, u32, u32) {
    let mut n1 = HashSet::<u16>::new();
    let mut n2 = HashSet::<u16>::new();
    let mut n3 = HashSet::<u16>::new();
    for node in trie.find_all_nodes(&rule) {
        for (key, child) in &node.children {
            match child.count {
                1 => { n1.insert(*key); },
                2 => { n2.insert(*key); },
                _ => { n3.insert(*key); }
            }
        }
    }
    let result = (n1.len() as u32, n2.len() as u32, n3.len() as u32);
    result
}