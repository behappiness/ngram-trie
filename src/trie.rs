pub mod trienode;

use trienode::TrieNode;
use serde::{Serialize, Deserialize};
use std::fs::{File, metadata};
use std::io::{BufReader, BufWriter};
use std::time::Instant;
use std::sync::Mutex;
use rclite::Arc;
use std::ops::Range;
use bincode::{serialize_into, deserialize_from};
use sorted_vector_map::{SortedVectorMap, SortedVectorSet};
use rayon::prelude::*;
use std::hash::{Hash, Hasher};
use lazy_static::lazy_static;
use quick_cache::sync::Cache;
use log::{info, error, debug};
use simple_tqdm::Tqdm;
const BATCH_SIZE: usize = 5_000_000;
const BATCH_ROOT_CAPACITY: usize = 0;

// the dataset size matters as well
const CACHE_SIZE_C: usize = 610*16384*32; //(rules+25%)*keys = RULES*KEYS
const CACHE_SIZE_N: usize = 610*3*32; //(rules+25%) = RULES*1.25

lazy_static! {
    pub static ref CACHE_C: Cache<Vec<Option<u16>>, u32> = Cache::new(CACHE_SIZE_C);
    pub static ref CACHE_N: Cache<Vec<Option<u16>>, Arc<Vec<Arc<TrieNode>>>> = Cache::new(CACHE_SIZE_N);
    pub static ref ZERO_COUNT_KEYS: Cache<u32, Arc<SortedVectorSet<u16>>> = Cache::new(3);
} 

#[derive(Serialize, Deserialize, Debug)]
pub struct NGramTrie {
    pub root: Arc<TrieNode>,
    pub n_gram_max_length: u32,
}

impl Default for NGramTrie {
    fn default() -> Self {
        NGramTrie::new(7, 2_usize.pow(14))
    }
}

impl NGramTrie {
    pub fn new(n_gram_max_length: u32, root_capacity: usize) -> Self {
        let mut node = TrieNode::new(Some(root_capacity));
        for i in 0..root_capacity {
            node.children.insert(i as u16, Arc::new(TrieNode::new(None)));
        }
        NGramTrie {
            root: Arc::new(node),
            n_gram_max_length
        }
    }

    #[inline]
    pub fn insert(&mut self, n_gram: &[u16]) {
        let root = Arc::get_mut(&mut self.root).unwrap();
        root.insert(n_gram);
    }

    #[inline]
    pub fn merge(&mut self, other: &NGramTrie) {
        info!("----- Merging tries -----");
        let start = Instant::now();
        let root = Arc::get_mut(&mut self.root).unwrap();
        root.merge(other.root.clone());
        let duration = start.elapsed();
        debug!("Time taken to merge tries: {:.2?}", duration);
    }

    pub fn shrink_to_fit(&mut self) {
        info!("----- Shrinking trie -----");
        let start = Instant::now();
        let root = Arc::get_mut(&mut self.root).unwrap();
        root.shrink_to_fit();
        let duration = start.elapsed();
        info!("Time taken to shrink to fit: {:.2?}", duration);
    }

    pub fn save(&self, filename: &str) {
        info!("----- Saving trie -----");
        let start = Instant::now();
        let _file = filename.to_owned() + ".trie";
        let file = File::create(&_file).unwrap_or_else(|e| {
            error!("Unable to create file {}: {}", &_file, e);
            panic!("Unable to create file");
        });
        let writer = BufWriter::new(file);
        if let Err(e) = serialize_into(writer, self) {
            error!("Serialization failed: {}", e);
            panic!("Serialization failed");
        }
        let duration = start.elapsed();
        info!("Time taken to save trie: {:.2?}", duration);
        let file_size = metadata(&_file).unwrap_or_else(|e| {
            error!("Unable to get file metadata {}: {}", &_file, e);
            panic!("Unable to get file metadata");
        }).len();
        let file_size_mb = file_size as f64 / (1024.0 * 1024.0);
        info!("Size of saved file: {:.2} MB", file_size_mb);
    }

    pub fn load(filename: &str) -> Self {
        info!("----- Loading trie -----");
        let start = Instant::now();
        let _file = filename.to_owned() + ".trie";
        let file = File::open(&_file).unwrap_or_else(|e| {
            error!("Unable to open file {}: {}", &_file, e);
            panic!("Unable to open file");
        });
        let reader = BufReader::new(file);
        let trie: NGramTrie = deserialize_from(reader).unwrap_or_else(|e| {
            error!("Deserialization failed: {}", e);
            panic!("Deserialization failed");
        });
        let duration = start.elapsed();
        info!("Time taken to load trie: {:.2?}", duration);
        trie
    }

    pub fn _preprocess_rule_context(tokens: &[u16], rule_context: Option<&str>) -> Vec<Option<u16>> {
        let mut result = Vec::new();
        if let Some(rule_context) = rule_context {
            let diff = tokens.len() - rule_context.len();
            for (&token, rule) in tokens[diff..].iter().zip(rule_context.chars()) {
                match rule {
                    '*' => result.push(None),
                    '-' => continue,
                    _ => result.push(Some(token)),
                }
            }
        } else {
            result = tokens.iter().map(|&t| Some(t)).collect();
        }
        result
    }

    pub fn _calculate_ruleset(n_gram_max_length: u32, characters: &[&str]) -> Vec<String> {
        if n_gram_max_length == 1 {
            return characters.iter().filter(|&&c| c != "*").map(|&c| c.to_string()).collect();
        }
        let mut ruleset = Vec::<String>::new();
        ruleset.extend(NGramTrie::_calculate_ruleset(n_gram_max_length - 1, characters));
    
        //let characters = vec!["+", "*", "-"];
        
        let mut combinations : Vec<String> = (2..n_gram_max_length).fold(
            characters.iter().map(|c| characters.iter().map(move |&d| d.to_owned() + *c)).flatten().collect(),
            |acc,_| acc.into_iter().map(|c| characters.iter().map(move |&d| d.to_owned() + &*c)).flatten().collect()
        );
    
        combinations.retain(|comb| comb.starts_with('+'));
    
        let mut tokens = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789".to_string();
        tokens.truncate(n_gram_max_length as usize);
        let mut hashmap = SortedVectorMap::<String, String>::new();
    
        for comb in combinations {
            let mut key = "".to_string();
            for (token, rule) in tokens.chars().zip(comb.chars()) {
                match rule {
                    '*' => key += "*",
                    '-' => continue,
                    _ => key += &token.to_string(),
                }
            }
            hashmap.insert(key, comb);
        }
    
        ruleset.extend(hashmap.values().cloned());
        ruleset.sort_by(|a, b| b.cmp(a));
        ruleset.sort_by(|a, b| a.len().cmp(&b.len()));
        ruleset
    }

    pub fn get_count(&self, rule: &[Option<u16>]) -> u32 {
        if let Some(cache) = CACHE_C.get(rule) {
            return cache;
        }

        let mut _count = 0;
        if let Some(cache) = CACHE_N.get(rule) {
            _count = cache.par_iter().map(|node| node.count).sum();
        } else if let Some(cache) = CACHE_N.get(&rule[..rule.len() - 1]) {
            _count = cache.par_iter().map(|node| node.get_count(&[rule[rule.len() - 1]])).sum();
        } else /*if !self.is_rule_in_zero_count_keys(rule)*/ {
            _count = self.root.get_count(rule);
        }

        CACHE_C.insert(rule.to_vec(), _count);
        _count
    }

    pub fn find_all_nodes(&self, rule: Vec<Option<u16>>) -> Arc<Vec<Arc<TrieNode>>> {
        let mut _nodes = Vec::new();
        if let Some(cache) = CACHE_N.get(&rule) {
            return cache.clone();
        } else if let Some(cache) = CACHE_N.get(&rule[..rule.len() - 1]) {
            _nodes = cache.par_iter().flat_map(|node| node.find_all_nodes(&[rule[rule.len() - 1]])).collect();
        } else /*if !self.is_rule_in_zero_count_keys(&rule)*/ {
            _nodes = self.root.find_all_nodes(&rule);
        }
        let nodes_arc = Arc::new(_nodes);
        CACHE_N.insert(rule, nodes_arc.clone());
        nodes_arc
    }

    pub fn cache_find_all_nodes(&self, history: &[u16], ruleset: &[String]) {
        for rule in ruleset {
            for i in (0..rule.len()).rev() {
                let _rule = Self::_preprocess_rule_context(&history, Some(&rule[i..].to_string()));
                self.find_all_nodes(_rule.clone());
                self.get_count(&_rule);
            }
        }
    }

    pub fn is_rule_in_zero_count_keys(&self, rule: &[Option<u16>]) -> bool {
        let keys = ZERO_COUNT_KEYS.get(&0).unwrap();
        rule.iter().any(|key| {
            if let Some(key) = key {
                keys.contains(&key)
            } else {
                false
            }
        })
    }

    pub fn estimate_time_and_ram(tokens_size: usize) -> (f64, f64) {
        let x = tokens_size as f64;
        let y = 0.0021 * x.powf(0.8525);
        let _x = (y / 0.0021).powf(1.0 / 0.8525) as f64; //how many can be fit in RAM
        let t = (2.8072 * x / 1_000_000.0 - 0.124) / 60.0; //how long it will take to fit
        info!("Expected time for {} tokens: {:.2} min", tokens_size, t);
        info!("Expected ram usage for {} tokens: {:.2} MB", tokens_size, y);
        (t, y)
    }
    
    pub fn fit(tokens: Arc<Vec<u16>>, n_gram_max_length: u32, root_capacity: usize, max_tokens: Option<usize>) -> Self {
        info!("----- Trie fitting -----");
        let tokens_size = max_tokens.unwrap_or(tokens.len());
        NGramTrie::estimate_time_and_ram(tokens_size);
        let mut trie = NGramTrie::new(n_gram_max_length, root_capacity);
        let max_tokens = max_tokens.unwrap_or(tokens.len()).min(tokens.len());
        let start = Instant::now();
        for i in (0..max_tokens - n_gram_max_length as usize + 1).tqdm() {
            trie.insert(&tokens[i..i + n_gram_max_length as usize]);
        }
        let duration = start.elapsed();
        info!("Time taken to fit trie: {:.2?}", duration);
        trie.shrink_to_fit();
        trie
    }

    #[deprecated]
    pub fn fit_multithreaded(tokens: Arc<Vec<u16>>, n_gram_max_length: u32, root_capacity: usize, max_tokens: Option<usize>) -> Self {
        info!("----- Trie fitting multithreaded -----");
        let root_trie = Arc::new(Mutex::new(NGramTrie::new(n_gram_max_length, root_capacity)));
        let tokens_size = max_tokens.unwrap_or(tokens.len());
        NGramTrie::estimate_time_and_ram(tokens_size);
        let batch_size = BATCH_SIZE;
        let num_batches = (tokens_size as f64 / batch_size as f64).ceil() as usize;

        let mut tries: Vec<(Self, Range<usize>)> = Vec::new();
        for batch in 0..num_batches {
            let batch_start = batch * batch_size;
            let batch_end = (batch_start + batch_size).min(tokens_size) - n_gram_max_length as usize + 1;
            let trie = NGramTrie::new(n_gram_max_length, BATCH_ROOT_CAPACITY);
            tries.push((trie, batch_start..batch_end));
        }

        let start = Instant::now();
        tries.par_iter_mut().for_each(|(trie, range)| {
            let start_fit = Instant::now();
            for i in range {
                trie.insert(&tokens[i..i + n_gram_max_length as usize]);
            }
            let duration_fit = start_fit.elapsed();
            debug!("Time taken to fit trie: {:.2?}", duration_fit);
            trie.shrink_to_fit();
            let mut root_trie = root_trie.lock().unwrap();
            root_trie.merge(trie);
        });
        let duration = start.elapsed();
        info!("Time taken to fit trie multithreaded: {:.2?}", duration);
        
        let mut root_trie = Arc::try_unwrap(root_trie).unwrap().into_inner().unwrap();
        root_trie.shrink_to_fit();
        root_trie
    }

    pub fn load_json(filename: &str, max_tokens: Option<usize>) -> std::io::Result<Arc<Vec<u16>>> {
        info!("----- Loading tokens -----");
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        let start = std::time::Instant::now();
        let mut tokens: Vec<u16> = serde_json::from_reader(reader).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let duration = start.elapsed();
        info!("Time taken to load tokens: {:.2?}", duration);
        debug!("Size of tokens in RAM before truncation: {:.2} MB", (tokens.len() * std::mem::size_of::<u16>()) as f64 / 1024.0 / 1024.0);
        if let Some(max) = max_tokens {
            if max < tokens.len() {
                tokens.truncate(max);
            }
        }
        info!("Size of tokens in RAM: {:.2} MB", (tokens.len() * std::mem::size_of::<u16>()) as f64 / 1024.0 / 1024.0);
        info!("Tokens loaded: {}", tokens.len());
        Ok(Arc::new(tokens))
    }
    
    pub fn init_cache(&self) {
        CACHE_C.insert(vec![], self.root.get_count(&vec![]));
        let mut zero_count_keys = SortedVectorSet::new();
        self.root.children.iter().for_each(|(key, child)| {
            if child.count == 0 {
                zero_count_keys.insert(*key);
            }
        });

        ZERO_COUNT_KEYS.insert(0, Arc::new(zero_count_keys));

        let nodes = vec![self.root.clone()];
        let nodes_arc = Arc::new(nodes);
        CACHE_N.insert(vec![], nodes_arc.clone());

    }

    pub fn reset_cache(&self) {
        info!("----- Resetting trie cache -----");
        CACHE_C.clear();
        CACHE_N.clear();
        self.init_cache();
    }

    pub fn count_nodes(&self) -> Vec<usize> {
        let mut counts = Vec::new();
        for i in 0..self.n_gram_max_length {
            counts.push(self.root.find_all_nodes(&vec![None; i as usize]).len());
        }
        counts
    }
}

impl Hash for NGramTrie {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.n_gram_max_length.hash(state);
        self.root.count.hash(state);
    }
}

impl PartialEq for NGramTrie {
    fn eq(&self, other: &Self) -> bool {
        self.n_gram_max_length == other.n_gram_max_length && self.root.count == other.root.count
    }
}

impl Eq for NGramTrie {}
