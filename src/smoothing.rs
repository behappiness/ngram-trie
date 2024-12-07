use crate::trie::NGramTrie;
use std::time::Instant;
use serde::{Serialize, Deserialize};
use rclite::Arc;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};
use quick_cache::sync::Cache;
use lazy_static::lazy_static;
use log::{info, debug};
use hashbrown::{HashSet, HashMap};

// the dataset size matters as well
const CACHE_SIZE_S: usize = 610*3*128; //(rules+25%) = RULES

lazy_static! {
    pub static ref CACHE_S: Cache<Vec<Option<u16>>, Arc<Vec<f64>>> = Cache::new(CACHE_SIZE_S);
}   

pub trait Smoothing: Sync + Send {
    fn smoothing(&self, trie: Arc<NGramTrie>, rule: &[Option<u16>]) -> Arc<Vec<f64>>;
    fn save(&self, filename: &str);
    fn load(&mut self, filename: &str);
    fn reset_cache(&self);
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ModifiedBackoffKneserNey {
    pub d1: Vec<f64>,
    pub d2: Vec<f64>,
    pub d3: Vec<f64>,
    pub unigram: Vec<f64>,
    pub vocabulary_size: usize,
    #[serde(skip)]
    pub trie: Arc<NGramTrie>
}

impl ModifiedBackoffKneserNey {
    pub fn new(trie: Arc<NGramTrie>) -> Self {
        let _vocabulary_size = trie.root.children.len();
        let (_d1, _d2, _d3, _uniform) = Self::calculate_d_values(trie.clone());
        ModifiedBackoffKneserNey {
            d1: _d1,
            d2: _d2,
            d3: _d3,
            unigram: Self::calculate_unigram_distribution(trie.clone()),
            vocabulary_size: _vocabulary_size,
            trie: trie
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
            // using root node so it doesn't cache anything
            trie.root.find_all_nodes(&vec![None; level as usize]).par_iter().for_each(|node| {
                match node.count {
                    1 => { n1.fetch_add(1, Ordering::Relaxed); },
                    2 => { n2.fetch_add(1, Ordering::Relaxed); },
                    3 => { n3.fetch_add(1, Ordering::Relaxed); },
                    4 => { n4.fetch_add(1, Ordering::Relaxed); },
                    _ => {}
                }
            });
            ns[level as usize - 1][0] = n1.load(Ordering::Relaxed);
            ns[level as usize - 1][1] = n2.load(Ordering::Relaxed);
            ns[level as usize - 1][2] = n3.load(Ordering::Relaxed);
            ns[level as usize - 1][3] = n4.load(Ordering::Relaxed);
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

    #[doc = "The unigram distribution \\( P_{\text{KN}}(w) \\) is used as the base case in this back-off process. It is calculated as:\n\
        \\[ \n\
        P_{\text{KN}}(w) = \\frac{\text{Number of unique bigrams ending in } w}{\text{Total number of unique bigrams}} \n\
        \\]"]
    pub fn calculate_unigram_distribution(trie: Arc<NGramTrie>) -> Vec<f64> {
        let mut continuation_counts: HashMap<u16, HashSet<u16>> = HashMap::new();
        let mut unigram: Vec<f64> = vec![0.0; trie.root.children.len()];

        trie.root.children.iter().for_each(|(first_key, child)| {
            child.children.iter().for_each(|(second_key, _)| {
                continuation_counts.entry(*second_key).or_insert_with(HashSet::new).insert(*first_key);
            });
        });

        // Calculate total number of unique bigrams
        let total_unique_bigrams: usize = continuation_counts.values().map(|set| set.len()).sum();

        // Calculate unigram distribution
        for (key, contexts) in continuation_counts.iter() {
            unigram[*key as usize] = contexts.len() as f64 / total_unique_bigrams as f64;
        }

        // Normalize the unigram array
        let sum: f64 = unigram.iter().sum();
        if sum > 0.0 {
        for value in unigram.iter_mut() {
                *value /= sum;
            }
        }
        debug!("Sum of unigrams: {:.4}", unigram.iter().sum::<f64>());
        unigram
    }

    pub fn calc_smoothed(&self, c_i: u32, c_i_minus_1: u32, level: usize, ns: (u32, u32, u32), s: f64) -> f64 {
        if c_i_minus_1 > 0 {
            let d = match c_i {
                0 => 0.0,
                1 => self.d1[level],
                2 => self.d2[level],
                _ => self.d3[level]
            };

            let alpha = (c_i as f64 - d).max(0.0) / c_i_minus_1 as f64;
            let gamma = (self.d1[level] * ns.0 as f64 + self.d2[level] * ns.1 as f64 + self.d3[level] * ns.2 as f64) / c_i_minus_1 as f64;
            alpha + gamma * s
        } else {
            s
        }
    }

    pub fn init_cache(&self) {
        // let nodes = self.trie.find_all_nodes(&vec![]);

        // let mut n1 = HashSet::<u16>::new();
        // let mut n2 = HashSet::<u16>::new();
        // let mut n3 = HashSet::<u16>::new();

        // let mut token_count_map: Vec<u32> = vec![0; self.vocabulary_size];
        // let mut result: Vec<f64> = vec![self.uniform; self.vocabulary_size];

        // nodes.iter().for_each(|node| {
        //     node.children.iter().for_each(|(key, child)| {
        //         match child.count { //maybe we have to sum over the keys and then do the match
        //             1 => { n1.insert(*key); },
        //             2 => { n2.insert(*key); },
        //             _ => { n3.insert(*key); }
        //         }
        //         token_count_map[*key as usize] += child.count;
        //     });
        // });

        // let c_i_minus_1 = token_count_map.iter().sum::<u32>();
        // let ns = (n1.len() as u32, n2.len() as u32, n3.len() as u32);

        // for i in 0..self.vocabulary_size {
        //     result[i] = self.calc_smoothed(token_count_map[i], c_i_minus_1, 0, ns, self.uniform);
        // }

        let _result = Arc::new(self.unigram.clone());

        CACHE_S.insert(vec![], _result);
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
        CACHE_S.clear();
        self.init_cache();
    }

    fn smoothing(&self, trie: Arc<NGramTrie>, rule: &[Option<u16>]) -> Arc<Vec<f64>> {
        if let Some(cached_value) = CACHE_S.get(rule) {
            return cached_value.clone();
        }

        let (token_count_map, c_i_minus_1, ns) = trie.get_token_count_map(&rule);
        let mut result: Vec<f64> = vec![0.0; token_count_map.len()];

        let s_map = self.smoothing(trie, &rule[..rule.len()-1]);

        //par doesnt help on small dataset so it wont help on big one (small array 16384)
        result.iter_mut().enumerate().for_each(|(i, res)| {
            *res = self.calc_smoothed(token_count_map[i], c_i_minus_1, rule.len(), ns, s_map[i]);
        });

        let _result = Arc::new(result);
        CACHE_S.insert(rule.to_vec(), _result.clone());
        _result
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct StupidBackoff {
    pub backoff_factor: f64,
    pub vocabulary_size: usize
}

impl StupidBackoff {
    pub fn new(trie: Arc<NGramTrie>, backoff_factor: Option<f64>) -> Self {
        StupidBackoff { backoff_factor: backoff_factor.unwrap_or(0.4), vocabulary_size: trie.root.children.len() as usize }
    }

    pub fn init_cache(&self) {
        CACHE_S.insert(vec![], Arc::new(vec![0.0; self.vocabulary_size]));
    }

    pub fn calc_stupid_backoff(&self, c_i: u32, c_i_minus_1: u32, s: f64) -> f64 {
        if c_i > 0 {
            c_i as f64 / c_i_minus_1 as f64
        } else {
            self.backoff_factor * s
        }
    }
}

impl Smoothing for StupidBackoff {
    fn smoothing(&self, trie: Arc<NGramTrie>, rule: &[Option<u16>]) -> Arc<Vec<f64>> {
        if let Some(cached_value) = CACHE_S.get(rule) {
            return cached_value.clone();
        }

        let (token_count_map, c_i_minus_1, _) = trie.get_token_count_map(&rule);
        let mut result: Vec<f64> = vec![0.0; token_count_map.len()];

        let s_map = self.smoothing(trie, &rule[..rule.len()-1]);

        //par doesnt help on small dataset so it wont help on big one (small array 16384)
        result.iter_mut().enumerate().for_each(|(i, res)| {
            *res = self.calc_stupid_backoff(token_count_map[i], c_i_minus_1, s_map[i]);
        });

        let _result = Arc::new(result);
        CACHE_S.insert(rule.to_vec(), _result.clone());
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
        CACHE_S.clear();
        self.init_cache();
    }
}


