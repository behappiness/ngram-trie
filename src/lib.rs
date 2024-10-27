pub mod trie;
pub mod smoothing;
pub mod smoothed_trie;

use pyo3::prelude::*;
use rclite::Arc;
use trie::NGramTrie;
use smoothed_trie::SmoothedTrie;
use pyo3_log;

#[pyclass]
struct PySmoothedTrie {
    smoothed_trie: SmoothedTrie,
}

#[pymethods]
impl PySmoothedTrie {
    #[new]
    #[pyo3(signature = (n_gram_max_length, root_capacity=None))]
    fn new(n_gram_max_length: u32, root_capacity: Option<usize>) -> Self {
        PySmoothedTrie {
            smoothed_trie: SmoothedTrie::new(NGramTrie::new(n_gram_max_length, root_capacity), None),
        }
    }

    fn save(&self, filename: &str) {
        self.smoothed_trie.save(filename);
    }

    fn load(&mut self, filename: &str) {
        self.smoothed_trie.load(filename);
    }

    fn reset_cache(&self) {
        self.smoothed_trie.reset_cache();
    }

    #[pyo3(signature = (tokens, n_gram_max_length, root_capacity=None, max_tokens=None, smoothing_name=None))]
    fn fit(&mut self, tokens: Vec<u16>, n_gram_max_length: u32, root_capacity: Option<usize>, max_tokens: Option<usize>, smoothing_name: Option<String>) {
        self.smoothed_trie.fit(Arc::new(tokens), n_gram_max_length, root_capacity, max_tokens, smoothing_name);
    }

    fn set_rule_set(&mut self, rule_set: Vec<String>) {
        self.smoothed_trie.set_rule_set(rule_set);
    }

    fn get_count(&self, rule: Vec<Option<u16>>) -> u32 {
        self.smoothed_trie.get_count(rule)
    }

    #[pyo3(signature = (smoothing_name=None))]
    fn fit_smoothing(&mut self, smoothing_name: Option<String>) {
        self.smoothed_trie.fit_smoothing(smoothing_name);
    }

    fn debug_cache_sizes(&self) {
        self.smoothed_trie.debug_cache_sizes();
    }

    fn get_prediction_probabilities(&self, history: Vec<u16>) -> Vec<(u16, Vec<(String, f64)>)> {
        self.smoothed_trie.get_prediction_probabilities(&history)
    }

    fn probability_for_token(&self, history: Vec<u16>, predict: u16) -> Vec<(String, f64)> {
        self.smoothed_trie.probability_for_token(&history, predict)
    }
}

#[pymodule]
fn ngram_trie(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();
    m.add_class::<PySmoothedTrie>()?;
    Ok(())
}
