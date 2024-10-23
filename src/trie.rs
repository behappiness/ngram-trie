pub mod trienode;

use trienode::TrieNode;
use crate::smoothing::Smoothing;
use serde::{Serialize, Deserialize};
use std::mem;
use std::fs::{File, metadata};
use std::io::{BufReader, BufWriter};
use std::time::Instant;
use std::sync::{Arc, Mutex};
use std::ops::Range;
use bincode::{serialize_into, deserialize_from};
use tqdm::tqdm;
use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;
use std::hash::{Hash, Hasher};
use lazy_static::lazy_static;
use quick_cache::sync::Cache;

const BATCH_SIZE: usize = 5_000_000;
const BATCH_ROOT_CAPACITY: usize = 0;
const CACHE_SIZE_C: usize = 32_000_000; //its related to the number of rules
const CACHE_SIZE_N: usize = 32_000_000; //its related to the number of rules, should be 233 for 7-grams

lazy_static! {
    static ref CACHE_C: Cache<Vec<Option<u16>>, u32> = Cache::new(CACHE_SIZE_C);
    static ref CACHE_N: Cache<Vec<Option<u16>>, Arc<Vec<TrieNode>>> = Cache::new(CACHE_SIZE_N);
} 

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct NGramTrie {
    pub root: Box<TrieNode>,
    pub n_gram_max_length: u32,
    pub rule_set: Vec<String>
}

impl Default for NGramTrie {
    fn default() -> Self {
        NGramTrie::new(7, None)
    }
}

impl NGramTrie {
    pub fn new(n_gram_max_length: u32, root_capacity: Option<usize>) -> Self {
        let _rule_set = NGramTrie::_calculate_ruleset(n_gram_max_length - 1);
        NGramTrie {
            root: Box::new(TrieNode::new(root_capacity)),
            n_gram_max_length,
            rule_set: _rule_set
        }
    }

    pub fn insert(&mut self, n_gram: &[u16]) {
        self.root.insert(n_gram);
    }

    pub fn merge(&mut self, other: &NGramTrie) {
        println!("----- Merging tries -----");
        let start = Instant::now();
        self.root.merge(&other.root);
        let duration = start.elapsed();
        println!("Time taken to merge tries: {:?}", duration);
    }

    pub fn size_in_ram(&self) -> usize {
        println!("----- Calculating size in RAM -----");
        let start = Instant::now();
        let size = mem::size_of::<NGramTrie>() + self.root.size_in_ram();
        let duration = start.elapsed();
        println!("Time taken to calculate size in RAM: {:?}", duration);
        println!("Size in RAM: {} MB", size as f64 / (1024.0 * 1024.0));
        size
    }

    pub fn shrink_to_fit(&mut self) {
        println!("----- Shrinking to fit -----");
        let start = Instant::now();
        self.root.shrink_to_fit();
        let duration = start.elapsed();
        println!("Time taken to shrink to fit: {:?}", duration);
    }

    pub fn save(&self, filename: &str) -> std::io::Result<()> {
        println!("----- Saving trie -----");
        let start = Instant::now();
        let _file = filename.to_owned() + ".trie";
        let file = File::create(&_file)?;
        let writer = BufWriter::new(file);
        serialize_into(writer, self).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let duration = start.elapsed();
        println!("Time taken to save trie: {:?}", duration);
        let file_size = metadata(&_file).expect("Unable to get file metadata").len();
        let file_size_mb = file_size as f64 / (1024.0 * 1024.0);
        println!("Size of saved file: {:.2} MB", file_size_mb);
        Ok(())
    }

    pub fn load(filename: &str) -> std::io::Result<Self> {
        println!("----- Loading trie -----");
        let start = Instant::now();
        let _file = filename.to_owned() + ".trie";
        let file = File::open(_file)?;
        let reader = BufReader::new(file);
        let trie: NGramTrie = deserialize_from(reader).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let duration = start.elapsed();
        println!("Time taken to load trie: {:?}", duration);
        trie.size_in_ram();
        Ok(trie)
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
        let mut hashmap = HashMap::<String, String>::new();
    
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
        ruleset.sort_by(|a, b| a.len().cmp(&b.len()));
        ruleset
    }

    pub fn set_rule_set(&mut self, rule_set: Vec<String>) {
        println!("----- Setting rule set -----");
        self.rule_set = rule_set;
        self.rule_set.sort_by(|a, b| a.len().cmp(&b.len()));
        println!("Rule set: {:?}", self.rule_set);
    }

    pub fn get_count(&self, rule: &[Option<u16>]) -> u32 {
        if let Some(cache) = CACHE_C.get(rule) {
            return cache;
        }

        if rule.is_empty() {
            return self.root.get_count(rule);
        }

        let mut count = 0;
        let rule_short = rule[..rule.len() - 1].to_vec();
        if let Some(cache) = CACHE_N.get(&rule_short) {
            count = cache.iter().map(|node| node.get_count(&[rule[rule.len() - 1]])).sum();
        } else {
            count = self.root.get_count(rule);
        }

        // let nodes = self.find_all_nodes(&rule[..rule.len() - 1]);
        // let count = nodes.iter().map(|node| node.get_count(&[rule[rule.len() - 1]])).sum();
        
        CACHE_C.insert(rule.to_vec(), count);
        count
    }

    //TODO: merge with unique_continuation_count?
    pub fn find_all_nodes(&self, rule: &[Option<u16>]) -> Arc<Vec<TrieNode>> {
        if let Some(cache) = CACHE_N.get(rule) {
            return cache.clone();
        }
        let nodes = self.root.find_all_nodes(rule);
        let nodes_clone: Vec<TrieNode> = nodes.iter().map(|&node| node.semi_deep_clone()).collect();
        let nodes_clone_arc = Arc::new(nodes_clone);
        CACHE_N.insert(rule.to_vec(), nodes_clone_arc.clone());
        nodes_clone_arc
    }

    // pub fn unique_continuations(&self, rule: &[Option<u16>]) -> HashSet<u16> {
    //     let mut unique = HashSet::<u16>::new();
    //     for node in self.find_all_nodes(rule) {
    //         unique.extend(node.children.keys());
    //     }
    //     unique
    // }

    pub fn estimate_time_and_ram(tokens_size: usize) -> (f64, f64) {
        let x = tokens_size as f64;
        let y = 0.0021 * x.powf(0.8525);
        let _x = (y / 0.0021).powf(1.0 / 0.8525) as f64; //how many can be fit in RAM
        let t = (2.8072 * x / 1_000_000.0 - 0.124) / 60.0; //how long it will take to fit
        println!("Expected time for {} tokens: {} min", tokens_size, t);
        println!("Expected ram usage for {} tokens: {} MB", tokens_size, y);
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
        println!("Time taken to fit trie: {:?}", duration);
        trie.shrink_to_fit();
        trie.size_in_ram();
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
            println!("Time taken to fit trie: {:?}", duration_fit);
            trie.shrink_to_fit();
            let mut root_trie = root_trie.lock().unwrap();
            root_trie.merge(trie);
        });
        let duration = start.elapsed();
        println!("Time taken to fit trie multithreaded: {:?}", duration);
        
        let mut root_trie = Arc::try_unwrap(root_trie).unwrap().into_inner().unwrap();
        root_trie.shrink_to_fit();
        root_trie.size_in_ram();
        root_trie
    }

    pub fn load_json(filename: &str, max_tokens: Option<usize>) -> std::io::Result<Arc<Vec<u16>>> {
        println!("----- Loading tokens -----");
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        let start = std::time::Instant::now();
        let mut tokens: Vec<u16> = serde_json::from_reader(reader).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let duration = start.elapsed();
        println!("Time taken to load tokens: {:?}", duration);
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