use crate::trie::NGramTrie;
use crate::smoothing::Smoothing;
use rclite::Arc;
use std::time::Instant;
use rayon::prelude::*;
use crate::smoothing::ModifiedBackoffKneserNey;
use simple_tqdm::ParTqdm;

pub struct SmoothedTrie {
    pub trie: Arc<NGramTrie>,
    pub smoothing: Box<dyn Smoothing>,
    pub rule_set: Vec<String>
}

impl SmoothedTrie {
    pub fn new(trie: NGramTrie, smoothing: Box<dyn Smoothing>) -> Self {
        let rule_set = NGramTrie::_calculate_ruleset(trie.n_gram_max_length);
        SmoothedTrie { trie: Arc::new(trie), smoothing: smoothing, rule_set: rule_set }
    }

    pub fn load(&mut self, filename: &str) {
        self.trie = Arc::new(NGramTrie::load(filename));
        self.smoothing.load(filename);
        self.reset_cache();
    }

    pub fn save(&self, filename: &str) {
        self.trie.save(filename);
        self.smoothing.save(filename);
    }

    pub fn reset_cache(&self) {
        self.trie.reset_cache();
        self.smoothing.reset_cache();
    }

    pub fn fit(&mut self, tokens: Arc<Vec<u16>>, n_gram_max_length: u32, root_capacity: Option<usize>, max_tokens: Option<usize>) {
        self.trie = Arc::new(NGramTrie::fit(tokens, n_gram_max_length, root_capacity, max_tokens));
        self.reset_cache();
    }

    pub fn set_rule_set(&mut self, rule_set: Vec<String>) {
        println!("----- Setting rule set -----");
        self.rule_set = rule_set;
        self.rule_set.sort_by(|a, b| b.cmp(a));
        self.rule_set.sort_by(|a, b| a.len().cmp(&b.len()));
        println!("Rule set: {:?}", self.rule_set.len());
    }

    pub fn fit_smoothing(&mut self) {
        self.smoothing.reset_cache();
        self.smoothing = Box::new(ModifiedBackoffKneserNey::new(self.trie.clone()));
    }

    pub fn get_count(&self, rule: Vec<Option<u16>>) -> u32 {
        self.trie.get_count(&rule)
    }

    pub fn probability_for_token(&self, history: &[u16], predict: u16) -> Vec<(String, f64)> {
        let mut rules_smoothed = Vec::<(String, f64)>::new();

        for r_set in &self.rule_set.iter().filter(|r| r.len() <= history.len()).collect::<Vec<_>>()[..] {
            let mut rule = NGramTrie::_preprocess_rule_context(history, Some(&r_set));
            rule.push(Some(predict));
            rules_smoothed.push((r_set.to_string(), self.smoothing.smoothing(self.trie.clone(), &rule)));
        }

        rules_smoothed
    }

    pub fn get_prediction_probabilities(&self, history: &[u16]) -> Vec<(u16, Vec<(String, f64)>)> { 
        println!("----- Getting prediction probabilities -----");
        let start = Instant::now();
        assert!(history.len() < self.trie.n_gram_max_length as usize, "History length must be less than the n-gram max length");
        let prediction_probabilities = self.trie.root.children.par_iter().tqdm()
            .map(|(token, _)| {
                let probabilities = self.probability_for_token(history, *token);
                (*token, probabilities)
            })
            .collect();

        let duration = start.elapsed();
        println!("Time taken to get prediction probabilities: {:.2?}", duration);

        prediction_probabilities
    }

}
