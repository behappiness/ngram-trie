pub mod trie;
pub mod smoothing;
pub mod smoothed_trie;

use pyo3::prelude::*;
use rclite::Arc;
use trie::NGramTrie;
use smoothed_trie::SmoothedTrie;
use pyo3_log;

#[pyclass]
#[doc = "A high-performance n-gram language model implementation using a trie-based data structure.\n\n\
         Supports various smoothing techniques and efficient storage/retrieval of n-grams."]
struct PySmoothedTrie {
    smoothed_trie: SmoothedTrie,
}

#[pymethods]
impl PySmoothedTrie {
    #[new]
    #[doc = "Initialize a new n-gram trie model.\n\n\
             Args:\n\
                 n_gram_max_length (int): Maximum length of n-grams to store\n\
                 root_capacity (Optional[int]): Initial capacity for the root node (for optimization)\n\n\
             Example:\n\
                 >>> trie = PySmoothedTrie(n_gram_max_length=3)"]
    #[pyo3(signature = (n_gram_max_length, root_capacity=None))]
    fn new(n_gram_max_length: u32, root_capacity: Option<usize>) -> Self {
        PySmoothedTrie {
            smoothed_trie: SmoothedTrie::new(NGramTrie::new(n_gram_max_length, root_capacity), None),
        }
    }

    #[doc = "Save the model to a file.\n\n\
             Args:\n\
                 filename (str): Path where to save the model. Example: 'model' -> model.trie & model.smoothing"]
    fn save(&self, filename: &str) {
        self.smoothed_trie.save(filename);
    }

    #[doc = "Load a model from a file.\n\n\
             Args:\n\
                 filename (str): Path to the saved model. Example: 'model' -> model.trie & model.smoothing"]
    fn load(&mut self, filename: &str) {
        self.smoothed_trie.load(filename);
    }

    #[doc = "Reset internal caches to free memory or if the model is changed"]
    fn reset_cache(&self) {
        self.smoothed_trie.reset_cache();
    }

    #[doc = "Fit the trie model with the given tokens.\n\n\
             Args:\n\
                 tokens (List[int]): List of token IDs (must be uint16)\n\
                 n_gram_max_length (int): Maximum length of n-grams to store\n\
                 root_capacity (Optional[int]): Initial capacity for optimization\n\
                 max_tokens (Optional[int]): Maximum number of tokens to use\n\
                 smoothing_name (Optional[str]): Smoothing method name (default: 'modified_kneser_ney')\n\n\
             Example:\n\
                 >>> trie.fit([1, 2, 3, 4, 5], n_gram_max_length=3)"]
    #[pyo3(signature = (tokens, n_gram_max_length, root_capacity=None, max_tokens=None, smoothing_name=None))]
    fn fit(&mut self, tokens: Vec<u16>, n_gram_max_length: u32, root_capacity: Option<usize>, max_tokens: Option<usize>, smoothing_name: Option<String>) {
        self.smoothed_trie.fit(Arc::new(tokens), n_gram_max_length, root_capacity, max_tokens, smoothing_name);
    }

    #[doc = "Set custom rules.\n\n\
             Args:\n\
                 rule_set (List[str]): List of smoothing rules. Example: ['+', '++', '+*-']"]
    fn set_rule_set(&mut self, rule_set: Vec<String>) {
        self.smoothed_trie.set_rule_set(rule_set);
    }


    #[doc = "Set all: +, *, - ruleset by n-gram length.\n\n\
             Args:\n\
                 rule_length (int): Length of ruleset to set"]
    fn set_all_ruleset_by_length(&mut self, rule_length: u32) {
        self.smoothed_trie.set_all_ruleset_by_length(rule_length);
    }

    #[doc = "Set suffix: + ruleset by n-gram length.\n\n\
             Args:\n\
                 rule_length (int): Length of ruleset to set"]
    fn set_suffix_ruleset_by_length(&mut self, rule_length: u32) {
        self.smoothed_trie.set_suffix_ruleset_by_length(rule_length);
    }

    #[doc = "Set subgram: +, - ruleset by n-gram length.\n\n\
             Args:\n\
                 rule_length (int): Length of ruleset to set"]
    fn set_subgram_ruleset_by_length(&mut self, rule_length: u32) {
        self.smoothed_trie.set_subgram_ruleset_by_length(rule_length);
    }

    #[doc = "Get count for a specific n-gram pattern.\n\n\
             Args:\n\
                 rule (List[Optional[int]]): List of token IDs (must be uint16). Example: [None, 1, 2]"]
    fn get_count(&self, rule: Vec<Option<u16>>) -> u32 {
        self.smoothed_trie.get_count(rule)
    }

    #[doc = "Fit the smoothing model with the given smoothing name.\n\n\
             Args:\n\
                 smoothing_name (Optional[str]): Smoothing method name (default: 'modified_kneser_ney')"]
    #[pyo3(signature = (smoothing_name=None))]
    fn fit_smoothing(&mut self, smoothing_name: Option<String>) {
        self.smoothed_trie.fit_smoothing(smoothing_name);
    }

    #[doc = "Writes cache sizes to the log (Debug level)"]
    fn debug_cache_sizes(&self) {
        self.smoothed_trie.debug_cache_sizes();
    }

    #[doc = "Get prediction probabilities for a given history.\n\n\
             Args:\n\
                 history (List[int]): List of token IDs (must be uint16)"]
    fn get_prediction_probabilities(&self, history: Vec<u16>) -> Vec<(u16, Vec<(String, f64)>)> {
        self.smoothed_trie.get_prediction_probabilities(&history)
    }

    #[doc = "Get probability distribution for a specific token.\n\n\
             Args:\n\
                 history (List[int]): List of token IDs (must be uint16)\n\
                 predict (int): Token ID to predict"]
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
