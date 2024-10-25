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
use tqdm::tqdm;
use sorted_vector_map::SortedVectorMap;
use rayon::prelude::*;
use std::hash::{Hash, Hasher};
use lazy_static::lazy_static;
use quick_cache::sync::Cache;

const BATCH_SIZE: usize = 5_000_000;
const BATCH_ROOT_CAPACITY: usize = 0;

// the dataset size matters as well
const CACHE_SIZE_C: usize = 233*16104*32; //(rules+25%)*keys = RULES*KEYS
const CACHE_SIZE_N: usize = 233*3*32; //(rules+25%) = RULES*1.25

lazy_static! {
    pub static ref CACHE_C: Cache<Vec<Option<u16>>, u32> = Cache::new(CACHE_SIZE_C);
    pub static ref CACHE_N: Cache<Vec<Option<u16>>, Arc<Vec<Arc<TrieNode>>>> = Cache::new(CACHE_SIZE_N);
} 

#[derive(Serialize, Deserialize, Debug)]
pub struct NGramTrie {
    pub root: Arc<TrieNode>,
    pub n_gram_max_length: u32
}

impl Default for NGramTrie {
    fn default() -> Self {
        NGramTrie::new(7, None)
    }
}

impl NGramTrie {
    pub fn new(n_gram_max_length: u32, root_capacity: Option<usize>) -> Self {
        NGramTrie {
            root: Arc::new(TrieNode::new(root_capacity)),
            n_gram_max_length
        }
    }

    pub fn insert(&mut self, n_gram: &[u16]) {
        let root = Arc::get_mut(&mut self.root).unwrap();
        root.insert(n_gram);
    }

    pub fn merge(&mut self, other: &NGramTrie) {
        println!("----- Merging tries -----");
        let start = Instant::now();
        let root = Arc::get_mut(&mut self.root).unwrap();
        root.merge(other.root.clone());
        let duration = start.elapsed();
        println!("Time taken to merge tries: {:.2?}", duration);
    }

    pub fn shrink_to_fit(&mut self) {
        println!("----- Shrinking to fit -----");
        let start = Instant::now();
        let root = Arc::get_mut(&mut self.root).unwrap();
        root.shrink_to_fit();
        let duration = start.elapsed();
        println!("Time taken to shrink to fit: {:.2?}", duration);
    }

    pub fn save(&self, filename: &str) {
        println!("----- Saving trie -----");
        let start = Instant::now();
        let _file = filename.to_owned() + ".trie";
        let file = File::create(&_file).expect("Unable to create file");
        let writer = BufWriter::new(file);
        serialize_into(writer, self).expect("Serialization failed");
        let duration = start.elapsed();
        println!("Time taken to save trie: {:.2?}", duration);
        let file_size = metadata(&_file).expect("Unable to get file metadata").len();
        let file_size_mb = file_size as f64 / (1024.0 * 1024.0);
        println!("Size of saved file: {:.2} MB", file_size_mb);
    }

    pub fn load(filename: &str) -> Self {
        println!("----- Loading trie -----");
        let start = Instant::now();
        let _file = filename.to_owned() + ".trie";
        let file = File::open(_file).expect("Unable to open file");
        let reader = BufReader::new(file);
        let trie: NGramTrie = deserialize_from(reader).expect("Deserialization failed");
        let duration = start.elapsed();
        println!("Time taken to load trie: {:.2?}", duration);
        trie
    }

    pub fn _preprocess_rule_context(tokens: &[u16], rule_context: Option<&str>) -> Vec<Option<u16>> {
        let mut result = Vec::new();
        if let Some(rule_context) = rule_context {
            assert!(tokens.len() >= rule_context.len(), "Tokens length must be at least as big as rule context length");
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

    pub fn _calculate_ruleset(n_gram_max_length: u32) -> Vec<String> {
        if n_gram_max_length == 1 {
            return vec!["+".to_string(), "-".to_string()];
        }
        let mut ruleset = Vec::<String>::new();
        ruleset.extend(NGramTrie::_calculate_ruleset(n_gram_max_length - 1));
    
        let characters = vec!["+", "*", "-"];
        
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

        // if rule.len() == 0 { return self.root.count; }

        let mut _count = 0;
        if let Some(cache) = CACHE_N.get(rule) {
            _count = cache.iter().map(|node| node.count).sum();
        } else if let Some(cache) = CACHE_N.get(&rule[..rule.len() - 1]) {
            _count = cache.iter().map(|node| node.get_count(&[rule[rule.len() - 1]])).sum();
        } else {
            _count = self.root.get_count(rule);
        }

        CACHE_C.insert(rule.to_vec(), _count);
        _count
    }

    pub fn find_all_nodes(&self, rule: Vec<Option<u16>>) -> Arc<Vec<Arc<TrieNode>>> {
        if let Some(cache) = CACHE_N.get(&rule) {
            return cache.clone();
        } else if let Some(cache) = CACHE_N.get(&rule[..rule.len() - 1]) {
            let nodes: Vec<Arc<TrieNode>> = cache.iter().flat_map(|node| node.find_all_nodes(&[rule[rule.len() - 1]])).collect();
            let nodes_arc = Arc::new(nodes);
            CACHE_N.insert(rule.to_vec(), nodes_arc.clone());
            return nodes_arc;
        }
        let nodes = self.root.find_all_nodes(&rule);
        let nodes_arc = Arc::new(nodes);
        CACHE_N.insert(rule.to_vec(), nodes_arc.clone());
        nodes_arc
    }

    pub fn estimate_time_and_ram(tokens_size: usize) -> (f64, f64) {
        let x = tokens_size as f64;
        let y = 0.0021 * x.powf(0.8525);
        let _x = (y / 0.0021).powf(1.0 / 0.8525) as f64; //how many can be fit in RAM
        let t = (2.8072 * x / 1_000_000.0 - 0.124) / 60.0; //how long it will take to fit
        println!("Expected time for {} tokens: {:.2} min", tokens_size, t);
        println!("Expected ram usage for {} tokens: {:.2} MB", tokens_size, y);
        (t, y)
    }
    
    pub fn fit(tokens: Arc<Vec<u16>>, n_gram_max_length: u32, root_capacity: Option<usize>, max_tokens: Option<usize>) -> Self {
        println!("----- Trie fitting -----");
        let tokens_size = max_tokens.unwrap_or(tokens.len());
        NGramTrie::estimate_time_and_ram(tokens_size);
        let mut trie = NGramTrie::new(n_gram_max_length, root_capacity);
        let max_tokens = max_tokens.unwrap_or(tokens.len()).min(tokens.len());
        let start = Instant::now();
        for i in tqdm(0..max_tokens - n_gram_max_length as usize + 1) {
            trie.insert(&tokens[i..i + n_gram_max_length as usize]);
        }
        let duration = start.elapsed();
        println!("Time taken to fit trie: {:.2?}", duration);
        trie.shrink_to_fit();
        trie
    }

    pub fn fit_multithreaded(tokens: Arc<Vec<u16>>, n_gram_max_length: u32, root_capacity: Option<usize>, max_tokens: Option<usize>) -> Self {
        println!("----- Trie fitting multithreaded -----");
        let root_trie = Arc::new(Mutex::new(NGramTrie::new(n_gram_max_length, root_capacity)));
        let tokens_size = max_tokens.unwrap_or(tokens.len());
        NGramTrie::estimate_time_and_ram(tokens_size);
        let batch_size = BATCH_SIZE;
        let num_batches = (tokens_size as f64 / batch_size as f64).ceil() as usize;

        let mut tries: Vec<(Self, Range<usize>)> = Vec::new();
        for batch in 0..num_batches {
            let batch_start = batch * batch_size;
            let batch_end = (batch_start + batch_size).min(tokens_size) - n_gram_max_length as usize + 1;
            let trie = NGramTrie::new(n_gram_max_length, Some(BATCH_ROOT_CAPACITY));
            tries.push((trie, batch_start..batch_end));
        }

        let start = Instant::now();
        tries.par_iter_mut().for_each(|(trie, range)| {
            let start_fit = Instant::now();
            for i in range {
                trie.insert(&tokens[i..i + n_gram_max_length as usize]);
            }
            let duration_fit = start_fit.elapsed();
            println!("Time taken to fit trie: {:.2?}", duration_fit);
            trie.shrink_to_fit();
            let mut root_trie = root_trie.lock().unwrap();
            root_trie.merge(trie);
        });
        let duration = start.elapsed();
        println!("Time taken to fit trie multithreaded: {:.2?}", duration);
        
        let mut root_trie = Arc::try_unwrap(root_trie).unwrap().into_inner().unwrap();
        root_trie.shrink_to_fit();
        root_trie
    }

    pub fn load_json(filename: &str, max_tokens: Option<usize>) -> std::io::Result<Arc<Vec<u16>>> {
        println!("----- Loading tokens -----");
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        let start = std::time::Instant::now();
        let mut tokens: Vec<u16> = serde_json::from_reader(reader).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let duration = start.elapsed();
        println!("Time taken to load tokens: {:.2?}", duration);
        println!("Size of tokens in RAM: {:.2} MB", (tokens.len() * std::mem::size_of::<u16>()) as f64 / 1024.0 / 1024.0);
        if let Some(max) = max_tokens {
            if max < tokens.len() {
                tokens.truncate(max);
            }
        }
        println!("Size of tokens in RAM after truncation: {:.2} MB", (tokens.len() * std::mem::size_of::<u16>()) as f64 / 1024.0 / 1024.0);
        println!("Tokens loaded: {}", tokens.len());
        Ok(Arc::new(tokens))
    }
    
    pub fn init_cache(&self) {
        CACHE_C.insert(vec![], self.root.get_count(&vec![]));
        let nodes = self.root.find_all_nodes(&vec![]);
        let nodes_arc = Arc::new(nodes);
        CACHE_N.insert(vec![], nodes_arc.clone());
    }

    pub fn reset_cache(&self) {
        CACHE_C.clear();
        CACHE_N.clear();
        self.init_cache();
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