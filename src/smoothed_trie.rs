use crate::trie::NGramTrie;
use crate::smoothing::Smoothing;
use crate::smoothing::CACHE_S_C;
use crate::smoothing::CACHE_S_N;
use crate::trie::CACHE_C;
use crate::trie::CACHE_N;
use rclite::Arc;
use std::time::Instant;
use rayon::prelude::*;
use crate::smoothing::ModifiedBackoffKneserNey;
use log::{info, debug, error};
use serde_json;
use std::fs;

pub struct SmoothedTrie {
    pub trie: Arc<NGramTrie>,
    pub smoothing: Box<dyn Smoothing>,
    pub rule_set: Vec<String>
}

impl SmoothedTrie {
    pub fn new(trie: NGramTrie, smoothing_name: Option<String>) -> Self {
        let rule_set = NGramTrie::_calculate_ruleset(trie.n_gram_max_length - 1, &["+", "*", "-"]);
        info!("Ruleset size: {}", rule_set.len());
        debug!("Ruleset: {:?}", rule_set);
        let trie = Arc::new(trie);
        let smoothing = match smoothing_name {
            Some(smoothing_name) => match smoothing_name.as_str() {
                "modified_kneser_ney" => Box::new(ModifiedBackoffKneserNey::new(trie.clone())),
                _ => Box::new(ModifiedBackoffKneserNey::new(trie.clone()))
            },
            None => Box::new(ModifiedBackoffKneserNey::new(trie.clone()))
        };
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

    pub fn fit(&mut self, tokens: Arc<Vec<u16>>, n_gram_max_length: u32, root_capacity: Option<usize>, max_tokens: Option<usize>, smoothing_name: Option<String>) {
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
        self.smoothing = match smoothing_name {
            Some(smoothing_name) => match smoothing_name.as_str() {
                "modified_kneser_ney" => Box::new(ModifiedBackoffKneserNey::new(self.trie.clone())),
                _ => Box::new(ModifiedBackoffKneserNey::new(self.trie.clone()))
            },
            None => Box::new(ModifiedBackoffKneserNey::new(self.trie.clone()))
        };
    }

    pub fn get_count(&self, rule: Vec<Option<u16>>) -> u32 {
        self.trie.get_count(&rule)
    }

    pub fn probability_for_token(&self, history: &[u16], predict: u16) -> Vec<(String, f64)> {
        let mut rules_smoothed = Vec::<(String, f64)>::new();

        //better to calculate these in order so we can utilize threads down the line better
        for r_set in &self.rule_set.iter().filter(|r| r.len() <= history.len()).collect::<Vec<_>>()[..] {
            let mut rule = NGramTrie::_preprocess_rule_context(history, Some(&r_set));
            rule.push(Some(predict));
            rules_smoothed.push((r_set.to_string(), self.smoothing.smoothing(self.trie.clone(), &rule)));
        }

        rules_smoothed
    }

    pub fn debug_cache_sizes(&self) {
        debug!("CACHE_S_C size: {}", CACHE_S_C.len());
        debug!("CACHE_S_N size: {}", CACHE_S_N.len());
        debug!("CACHE_C size: {}", CACHE_C.len());
        debug!("CACHE_N size: {}", CACHE_N.len());
    }

    pub fn get_prediction_probabilities(&self, history: &[u16]) -> Vec<(u16, Vec<(String, f64)>)> { 
        info!("----- Getting prediction probabilities -----");
        let start = Instant::now();
        if history.len() >= self.trie.n_gram_max_length as usize {
            error!("History length must be less than the n-gram max length");
            panic!("History length must be less than the n-gram max length");
        }
        let _asd = self.probability_for_token(history, history[0]);
        let prediction_probabilities = self.trie.root.children.par_iter() //.tqdm()
            .map(|(token, _)| {
                let probabilities = self.probability_for_token(history, *token);
                (*token, probabilities)
            })
            .collect();

        let duration = start.elapsed();
        info!("Time taken to get prediction probabilities: {:.2?}", duration);

        prediction_probabilities
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
}
