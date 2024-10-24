use crate::trie::NGramTrie;
use crate::smoothing::Smoothing;
use std::sync::Arc;
use std::time::Instant;
use rayon::prelude::*;
use crate::smoothing::ModifiedBackoffKneserNey;
use simple_tqdm::ParTqdm;
use tqdm::Iter;

pub struct SmoothedTrie {
    pub trie: Arc<NGramTrie>,
    pub smoothing: Box<dyn Smoothing>
}

impl SmoothedTrie {
    pub fn new(trie: NGramTrie, smoothing: Box<dyn Smoothing>) -> Self {
        SmoothedTrie { trie: Arc::new(trie), smoothing: smoothing }
    }

    pub fn load(&mut self, filename: &str) {
        self.trie = Arc::new(NGramTrie::load(filename).unwrap());
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
        let mut trie: NGramTrie = (*self.trie).clone();
        trie.rule_set = rule_set;
        self.trie = Arc::new(trie);
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
        let _trie = self.trie.clone();

        for r_set in &_trie.rule_set.iter().filter(|r| r.len() <= history.len()).collect::<Vec<_>>()[..] {
            let mut rule = NGramTrie::_preprocess_rule_context(history, Some(&r_set));
            rule.push(Some(predict));
            rules_smoothed.push((r_set.to_string(), self.smoothing.smoothing(self.trie.clone(), &rule)));
        }

        rules_smoothed
    }

    pub fn get_prediction_probabilities(&self, history: &[u16]) -> Vec<(u16, Vec<(String, f64)>)> { 
        println!("----- Getting prediction probabilities -----");
        let start = Instant::now();
        let _trie = self.trie.clone();
        let prediction_probabilities = _trie.root.children.par_iter().tqdm()
            .map(|(token, _)| {
                let probabilities = self.probability_for_token(history, *token);
                (*token, probabilities)
            })
            .collect();

        let duration = start.elapsed();
        println!("Time taken to get prediction probabilities: {:?}", duration);

        prediction_probabilities
    }

}
