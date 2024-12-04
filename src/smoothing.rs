use crate::trie::NGramTrie;
use std::time::Instant;
use serde::{Serialize, Deserialize};
use rclite::Arc;
use rayon::{prelude::*, vec};
use std::sync::atomic::{AtomicU32, Ordering};
use quick_cache::sync::Cache;
use lazy_static::lazy_static;
use dashmap::DashSet;
use log::{info, debug};

// the dataset size matters as well
const CACHE_SIZE_S_C: usize = 610*16384*32; //(rules+25%)*keys = RULES*KEYS
const CACHE_SIZE_S_N: usize = 610*3*32; //(rules+25%) = RULES*1.25

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
    pub d1: Vec<f64>,
    pub d2: Vec<f64>,
    pub d3: Vec<f64>,
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

    pub fn calculate_d_values(trie: Arc<NGramTrie>) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64) {
        let mut d1 = vec![0.0; trie.n_gram_max_length as usize];
        let mut d2 = vec![0.0; trie.n_gram_max_length as usize];
        let mut d3 = vec![0.0; trie.n_gram_max_length as usize];
        if trie.root.children.len() == 0 {
            return (d1, d2, d3, 0.0);
        }
        info!("----- Calculating d values for smoothing -----");
        let start = Instant::now();
        let mut ns = vec![vec![0 as u32; 4]; trie.n_gram_max_length as usize];
        for level in 1..=trie.n_gram_max_length {
            let n1 = Arc::new(AtomicU32::new(0));
            let n2 = Arc::new(AtomicU32::new(0));
            let n3 = Arc::new(AtomicU32::new(0));
            let n4 = Arc::new(AtomicU32::new(0));
            trie.find_all_nodes(vec![None; level as usize]).par_iter().for_each(|node| {
                match node.count {
                    1 => { n1.fetch_add(1, Ordering::SeqCst); },
                    2 => { n2.fetch_add(1, Ordering::SeqCst); },
                    3 => { n3.fetch_add(1, Ordering::SeqCst); },
                    4 => { n4.fetch_add(1, Ordering::SeqCst); },
                    _ => {}
                }
            });
            ns[level as usize - 1][0] = n1.load(Ordering::SeqCst);
            ns[level as usize - 1][1] = n2.load(Ordering::SeqCst);
            ns[level as usize - 1][2] = n3.load(Ordering::SeqCst);
            ns[level as usize - 1][3] = n4.load(Ordering::SeqCst);
        }

        let uniform = 1.0 / trie.root.children.len() as f64;

        for i in 0..trie.n_gram_max_length as usize {
            if ns[i][0] == 0 || ns[i][1] == 0 || ns[i][2] == 0 || ns[i][3] == 0 {
                d1[i] = 0.1;
                d2[i] = 0.2;
                d3[i] = 0.3;
            } else {
                let y = ns[i][0] as f64 / (ns[i][0] as f64 + 2.0 * ns[i][1] as f64);
                d1[i] = 1.0 - 2.0 * y * (ns[i][1] as f64 / ns[i][0] as f64);
                d2[i] = 2.0 - 3.0 * y * (ns[i][2] as f64 / ns[i][1] as f64);
                d3[i] = 3.0 - 4.0 * y * (ns[i][3] as f64 / ns[i][2] as f64);
            }
        }

        let elapsed = start.elapsed();
        info!("Time taken: {:.2?}", elapsed);
        info!("Smoothing calculated, d1: {:?}, d2: {:?}, d3: {:?}, uniform: {:.4}", d1, d2, d3, uniform);
        (d1, d2, d3, uniform)
    }

    pub fn count_unique_ns(trie: Arc<NGramTrie>, rule: Vec<Option<u16>>) -> (u32, u32, u32) {
        if let Some(cached_value) = CACHE_S_N.get(&rule) {
            return cached_value;
        }
        let n1 = DashSet::<u16>::new();
        let n2 = DashSet::<u16>::new();
        let n3 = DashSet::<u16>::new();
        trie.find_all_nodes(rule.clone()).iter().for_each(|node| {
            node.children.iter().for_each(|(key, child)| {
                match child.count { //maybe we have to sum over the keys and then do the match
                    1 => { n1.insert(*key); },
                    2 => { n2.insert(*key); },
                    _ => { n3.insert(*key); }
                }
            });
        });
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
        info!("----- Saving smoothing to file -----");
        let _file = filename.to_owned() + "_smoothing.json";
        let serialized = serde_json::to_string(self).unwrap();
        std::fs::write(_file, serialized).unwrap();
    }

    fn load(&mut self, filename: &str) {
        info!("----- Loading smoothing from file -----");
        let _file = filename.to_owned() + "_smoothing.json";
        let serialized = std::fs::read_to_string(_file).unwrap();
        *self = serde_json::from_str(&serialized).unwrap();
    }

    fn reset_cache(&self) {
        info!("----- Resetting smoothing cache -----");
        CACHE_S_C.clear();
        CACHE_S_N.clear();
        self.init_cache();
    }

    fn smoothing(&self, trie: Arc<NGramTrie>, rule: &[Option<u16>]) -> f64 {
        if let Some(cached_value) = CACHE_S_C.get(rule) {
            return cached_value;
        }

        //let w_i = &rule[rule.len() - 1];
        let w_i_minus_1 = &rule[..rule.len() - 1];

        let c_i_minus_1 = trie.get_count(&w_i_minus_1);
        
        let result = if c_i_minus_1 > 0 {
            let (n1, n2, n3) = Self::count_unique_ns(trie.clone(), w_i_minus_1.to_vec());

            let c_i = trie.get_count(&rule);

            let level = trie.n_gram_max_length as usize - rule.len();

            let d = match c_i {
                0 => 0.0,
                1 => self.d1[level],
                2 => self.d2[level],
                _ => self.d3[level]
            };

            let alpha = (c_i as f64 - d).max(0.0) / c_i_minus_1 as f64;
            if n1 == 0 && n2 == 0 && n3 == 0 {
                alpha
            } else {
                let gamma = (self.d1[level] * n1 as f64 + self.d2[level] * n2 as f64 + self.d3[level] * n3 as f64) / c_i_minus_1 as f64;

                alpha + gamma * self.smoothing(trie.clone(), &rule[1..])
            }
        } else {
            self.smoothing(trie.clone(), &rule[1..])
        };
        CACHE_S_C.insert(rule.to_vec(), result);
        result
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct StupidBackoff {
    pub backoff_factor: f64,
}

impl StupidBackoff {
    pub fn new(backoff_factor: Option<f64>) -> Self {
        StupidBackoff { backoff_factor: backoff_factor.unwrap_or(0.4) }
    }

    pub fn init_cache(&self) {
        CACHE_S_C.insert(vec![], 0.0);
    }
}

impl Smoothing for StupidBackoff {
    fn smoothing(&self, trie: Arc<NGramTrie>, rule: &[Option<u16>]) -> f64 {
        if let Some(cached_value) = CACHE_S_C.get(rule) {
            return cached_value;
        }
        let c_i = trie.get_count(&rule);
        let mut _result = 0.0;
        if c_i > 0 {
            let c_i_minus_1 = trie.get_count(&rule[..rule.len() - 1]); //cannot be 0 if higher order ngrams is non-zero
            _result = c_i as f64 / c_i_minus_1 as f64;
        } else if rule.len() > 1 {
            _result = self.backoff_factor * self.smoothing(trie, &rule[1..]);
        } else {
            _result = 0.0;
        }
        CACHE_S_C.insert(rule.to_vec(), _result);
        _result
    }

    fn save(&self, filename: &str) {
        info!("----- Saving stupid backoff to file -----");
        let _file = filename.to_owned() + "_stupid_backoff.json";
        let serialized = serde_json::to_string(self).unwrap();
        std::fs::write(_file, serialized).unwrap();
    }

    fn load(&mut self, filename: &str) {
        info!("----- Loading stupid backoff from file -----");
        let _file = filename.to_owned() + "_stupid_backoff.json";
        let serialized = std::fs::read_to_string(_file).unwrap();
        *self = serde_json::from_str(&serialized).unwrap();
    }

    fn reset_cache(&self) {
        info!("----- Resetting stupid backoff cache -----");
        CACHE_S_C.clear();
        CACHE_S_N.clear();
        self.init_cache();
    }
}


