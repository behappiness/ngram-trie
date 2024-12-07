#![allow(warnings)]
use std::{fs, time::Instant};

use crate::trie::*;
use rclite::Arc;
use simple_tqdm::Tqdm;
use crate::smoothing::*;
use log::{info, debug, error};
use serde_json;

use rayon::ThreadPoolBuilder;


pub struct SmoothedTrie {
    pub trie: Arc<NGramTrie>,
    pub smoothing: Box<dyn Smoothing>,
    pub rule_set: Vec<String>
}

impl SmoothedTrie {
    pub fn new(trie: NGramTrie, smoothing_name: Option<String>) -> Self {
        info!("----- Configuring number of threads to use for parallel operations -----");

        let num_threads = std::thread::available_parallelism()
            .map(|p| p.get().saturating_sub(2))
            .unwrap_or(1);
        
        ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .unwrap();

        let rule_set = NGramTrie::_calculate_ruleset(trie.n_gram_max_length - 1, &["+", "*", "-"]);
        info!("Ruleset size: {}", rule_set.len());
        debug!("Ruleset: {:?}", rule_set);
        info!("----- Creating smoothed trie -----");
        let trie = Arc::new(trie);
        let smoothing = Self::choose_smoothing(trie.clone(), smoothing_name);
        SmoothedTrie { trie: trie, smoothing: smoothing, rule_set: rule_set }
    }

    pub fn load(&mut self, filename: &str) {
        self.trie = Arc::new(NGramTrie::load(filename));
        self.smoothing.load(filename);

        // Load the ruleset from a JSON file
        let ruleset_file = format!("{}_ruleset.json", filename);
        let contents = fs::read_to_string(&ruleset_file).expect("Unable to read ruleset file");
        self.rule_set = serde_json::from_str(&contents).expect("Unable to parse ruleset");

        self.reset_cache();
    }

    pub fn save(&self, filename: &str) {
        self.trie.save(filename);
        self.smoothing.save(filename);

        // Save the ruleset to a JSON file
        let serialized_ruleset = serde_json::to_string(&self.rule_set).expect("Unable to serialize ruleset");
        let ruleset_file = format!("{}_ruleset.json", filename);
        fs::write(&ruleset_file, serialized_ruleset).expect("Unable to write ruleset file");
    }

    pub fn reset_cache(&self) {
        self.trie.reset_cache();
        self.smoothing.reset_cache();
    }

    pub fn fit(&mut self, tokens: Arc<Vec<u16>>, n_gram_max_length: u32, root_capacity: usize, max_tokens: Option<usize>, smoothing_name: Option<String>) {
        self.trie = Arc::new(NGramTrie::fit(tokens, n_gram_max_length, root_capacity, max_tokens));
        self.set_rule_set(NGramTrie::_calculate_ruleset(n_gram_max_length - 1, &["+", "*", "-"]));
        self.fit_smoothing(smoothing_name);
    }

    pub fn set_rule_set(&mut self, rule_set: Vec<String>) {
        info!("----- Setting ruleset -----");
        self.rule_set = rule_set;
        self.rule_set.sort_by(|a, b| b.cmp(a));
        self.rule_set.sort_by(|a, b| a.len().cmp(&b.len()));
        info!("Ruleset size: {}", self.rule_set.len());
        debug!("Ruleset: {:?}", self.rule_set);
    }

    pub fn fit_smoothing(&mut self, smoothing_name: Option<String>) {
        self.reset_cache();
        self.smoothing = Self::choose_smoothing(self.trie.clone(), smoothing_name);
    }

    pub fn choose_smoothing(trie: Arc<NGramTrie>, smoothing_name: Option<String>) -> Box<dyn Smoothing> {
        match smoothing_name {
            Some(smoothing_name) => match smoothing_name.as_str() {
                "modified_kneser_ney" => Box::new(ModifiedBackoffKneserNey::new(trie.clone())),
                "stupid_backoff" => Box::new(StupidBackoff::new(trie.clone(), None)),
                _ => Box::new(ModifiedBackoffKneserNey::new(trie.clone()))
            },
            None => Box::new(ModifiedBackoffKneserNey::new(trie.clone()))
        }
    }

    pub fn get_count(&self, rule: Vec<Option<u16>>) -> u32 {
        self.trie.get_count(&rule)
    }

    pub fn debug_cache_sizes(&self) {
        debug!("CACHE_S size: {}", CACHE_S.len());
        debug!("CACHE_C size: {}", CACHE_C.len());
        debug!("CACHE_N size: {}", CACHE_N.len());
    }

    pub fn get_smoothed_probabilities(&self, history: &[u16], rule_set: Option<Vec<String>>) -> Vec<(String, Vec<f64>)> { 
        info!("----- Getting smoothed probabilities -----");
        let start = Instant::now();
        if history.len() >= self.trie.n_gram_max_length as usize {
            error!("History length must be less than the n-gram max length");
            panic!("History length must be less than the n-gram max length");
        }
        let mut rule_set = if let Some(rs) = rule_set {
            let mut rs = rs;
            rs.sort_by(|a, b| b.cmp(a));
            rs.sort_by(|a, b| a.len().cmp(&b.len()));
            rs
        } else {
            self.rule_set.clone()
        };

        let mut smoothed_probabilities: Vec<(String, Vec<(f64)>)> = rule_set.iter().filter(|r| r.len() <= history.len()).map(|r| {
            let rule = NGramTrie::_preprocess_rule_context(history, Some(&r));
            (r.to_string(), self.smoothing.smoothing(self.trie.clone(), &rule).to_vec())
        }).collect();

        let duration = start.elapsed();
        info!("Time taken to get smoothed probabilities: {:.2?}", duration);

        // Normalize the probabilities for every rule
        // for (_, tokens) in &mut smoothed_probabilities {
        //     let total_prob: f64 = tokens.iter().map(|(prob)| prob).sum();
        //     for (prob) in tokens.iter_mut() {
        //         *prob /= total_prob;
        //     }
        // }

        smoothed_probabilities
    }

    pub fn get_unsmoothed_probabilities(&self, history: &[u16]) -> Vec<(String, Vec<(u16, f64)>)> {

        let mut rules_unsmoothed = Vec::<(String, Vec<(u16, f64)>)>::new();

        for r_set in &self.rule_set.iter().filter(|r| r.len() <= history.len()).collect::<Vec<_>>()[..] {
            let rule = NGramTrie::_preprocess_rule_context(history, Some(&r_set));
            let matches = self.trie.find_all_nodes(&rule);
            
            // Use a HashMap to aggregate counts for same tokens
            let token_count_map = matches.iter()
                .flat_map(|node| node.children.iter())
                .fold(std::collections::HashMap::new(), |mut map, (&token, child)| {
                    *map.entry(token).or_insert(0) += child.count;
                    map
                });

            // Convert HashMap to Vec and sort by token
            let mut token_counts: Vec<(u16, u32)> = token_count_map.into_iter().collect();
            token_counts.sort_by_key(|&(token, _)| token);

            let total_count: u32 = token_counts.iter().map(|(_, count)| count).sum();
            let token_probs: Vec<(u16, f64)> = token_counts.into_iter()
                .map(|(token, count)| (token, count as f64 / total_count as f64))
                .collect();
                
            rules_unsmoothed.push((r_set.to_string(), token_probs));
        }
        
        rules_unsmoothed
    }

    pub fn set_all_ruleset_by_length(&mut self, rule_length: u32) {
        let rule_set = NGramTrie::_calculate_ruleset(rule_length, &["+", "*", "-"]);
        self.set_rule_set(rule_set);
    }

    pub fn set_suffix_ruleset_by_length(&mut self, rule_length: u32) {
        let rule_set = NGramTrie::_calculate_ruleset(rule_length, &["+"]);
        self.set_rule_set(rule_set);
    }

    pub fn set_subgram_ruleset_by_length(&mut self, rule_length: u32) {
        let rule_set = NGramTrie::_calculate_ruleset(rule_length, &["+", "-"]);
        self.set_rule_set(rule_set);
    }

    pub fn count_nodes(&self) -> Vec<usize> {
        self.trie.count_nodes()
    }

    pub fn average_branching_factor_per_layer(&self) -> Vec<f64> {
        self.trie.average_branching_factor_per_layer()
    }
}
