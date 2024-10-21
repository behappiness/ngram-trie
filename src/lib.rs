mod trie;

use pyo3::prelude::*;
use std::sync::Arc;
use trie::NGramTrie;
use trie::smoothing::ModifiedBackoffKneserNey;

#[pyclass]
struct PyNGramTrie {
    trie: NGramTrie,
}

#[pymethods]
impl PyNGramTrie {
    #[new]
    #[pyo3(signature = (n_gram_max_length, root_capacity=None))]
    fn new(n_gram_max_length: u32, root_capacity: Option<usize>) -> Self {
        PyNGramTrie {
            trie: NGramTrie::new(n_gram_max_length, root_capacity),
        }
    }

    fn save(&self, filename: &str) {
        self.trie.save(filename);
    }

    fn load(&mut self, filename: &str) -> Result<(), std::io::Error> {
        self.trie = NGramTrie::load(filename)?;
        Ok(())
    }

    #[pyo3(signature = (tokens, n_gram_max_length, root_capacity=None, max_tokens=None))]
    fn fit(&mut self, tokens: Vec<u16>, n_gram_max_length: u32, root_capacity: Option<usize>, max_tokens: Option<usize>) {
        self.trie = NGramTrie::fit(Arc::new(tokens), n_gram_max_length, root_capacity, max_tokens);
    }

    fn get_prediction_probabilities(&self, smoothing: &PyModifiedBackoffKneserNey, history: Vec<u16>) -> Vec<(u16, Vec<(String, f64)>)> {
        self.trie.get_prediction_probabilities(&smoothing.smoothing, &history)
    }

    fn size_in_ram(&self) -> usize {
        self.trie.size_in_ram()
    }

    fn set_rule_set(&mut self, rule_set: Vec<String>) {
        self.trie.set_rule_set(rule_set);
    }

    fn get_count(&self, rule: Vec<Option<u16>>) -> u32 {
        self.trie.get_count(&rule)
    }
}

#[pyclass]
struct PyModifiedBackoffKneserNey {
    smoothing: ModifiedBackoffKneserNey,
}

#[pymethods]
impl PyModifiedBackoffKneserNey {
    #[new]
    fn new(trie: &PyNGramTrie) -> Self {
        PyModifiedBackoffKneserNey {
            smoothing: ModifiedBackoffKneserNey::new(&trie.trie),
        }
    }

    fn save(&self, filename: &str) {
        self.smoothing.save(filename);
    }

    #[staticmethod]
    fn load(filename: &str) -> Self {
        PyModifiedBackoffKneserNey { smoothing: ModifiedBackoffKneserNey::load(filename) }
    }
}

#[pymodule]
fn ngram_trie(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNGramTrie>()?;
    m.add_class::<PyModifiedBackoffKneserNey>()?;
    Ok(())
}
