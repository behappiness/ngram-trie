# ngram-trie

`ngram-trie` is a Rust library designed to efficiently handle n-gram data structures using a trie-based approach. It provides functionalities for fitting, saving, loading, and querying n-gram models, with support for various smoothing techniques.

## Installation Rust

1. Include it in the Cargo.toml:

    ```toml
    [dependencies]
    ngram-trie = { git = "https://github.com/behappiness/ngram-trie" }
    ```

## Installation Python

1. Install from pip:

    ```bash
    pip install ngram-trie
    ```


## Example Usage
```python
from ngram_trie import PySmoothedTrie

trie = PySmoothedTrie(n_gram_max_length=7, root_capacity=None)

trie.fit(tokenized_data, n_gram_max_length=7, root_capacity=None, max_tokens=None)

trie.set_rule_set(["++++++", "+++++", "++++", "+++", "++", "+"])

trie.fit_smoothing()

trie.get_prediction_probabilities(tokenized_context)
```

#### Specify the smoothing

```python
trie.fit_smoothing("modified_kneser_ney"/"stupid_backoff")
```

#### Unsmoothed

```python
from ngram_trie import PySmoothedTrie

trie = PySmoothedTrie(n_gram_max_length=7, root_capacity=None)

trie.fit(tokenized_data, n_gram_max_length=7, root_capacity=None, max_tokens=None)

trie.set_rule_set(rules)

trie.get_unsmoothed_probabilities(tokenized_context)
```

## Dev
```bash
cargo add pyo3 --features extension-module
```

#### Build wheel
```bash
maturin build
```


