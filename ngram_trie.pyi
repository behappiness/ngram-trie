from typing import List, Optional, Tuple

class PySmoothedTrie:
    """
    A high-performance n-gram language model implementation using a trie-based data structure.

    Supports various smoothing techniques and efficient storage/retrieval of n-grams.
    """

    def __init__(self, n_gram_max_length: int, root_capacity: int) -> None:
        """
        Initialize a new n-gram trie model.

        Args:
            n_gram_max_length (int): Maximum length of n-grams to store
            root_capacity (int): Initial capacity (tokenizer size) for the root node (for optimization)

        Example:
            >>> trie = PySmoothedTrie(n_gram_max_length=3, root_capacity=2**14)
        """
        ...

    def save(self, filename: str) -> None:
        """
        Save the model to a file.

        Args:
            filename (str): Path where to save the model. Example: 'model' -> model.trie & model.smoothing
        """
        ...

    def load(self, filename: str) -> None:
        """
        Load a model from a file.

        Args:
            filename (str): Path to the saved model. Example: 'model' -> model.trie & model.smoothing
        """
        ...

    def reset_cache(self) -> None:
        """
        Reset internal caches to free memory or if the model is changed.
        """
        ...

    def fit(self, tokens: List[int], n_gram_max_length: int, root_capacity: Optional[int] = None, max_tokens: Optional[int] = None, smoothing_name: Optional[str] = None) -> None:
        """
        Fit the trie model with the given tokens.

        Args:
            tokens (List[int]): List of token IDs (must be uint16)
            n_gram_max_length (int): Maximum length of n-grams to store
            root_capacity (Optional[int]): Initial capacity for optimization
            max_tokens (Optional[int]): Maximum number of tokens to use
            smoothing_name (Optional[str]): Smoothing method name (default: 'modified_kneser_ney')

        Example:
            >>> trie.fit([1, 2, 3, 4, 5], n_gram_max_length=3)
        """
        ...

    def set_rule_set(self, rule_set: List[str]) -> None:
        """
        Set custom rules.

        Args:
            rule_set (List[str]): List of smoothing rules. Example: ['+', '++', '+*-']
        """
        ...

    def set_all_ruleset_by_length(self, rule_length: int) -> None:
        """
        Set all: +, *, - ruleset by n-gram length.

        Args:
            rule_length (int): Length of ruleset to set
        """
        ...

    def set_suffix_ruleset_by_length(self, rule_length: int) -> None:
        """
        Set suffix: + ruleset by n-gram length.

        Args:
            rule_length (int): Length of ruleset to set
        """
        ...

    def set_subgram_ruleset_by_length(self, rule_length: int) -> None:
        """
        Set subgram: +, - ruleset by n-gram length.

        Args:
            rule_length (int): Length of ruleset to set
        """
        ...

    def get_count(self, rule: List[Optional[int]]) -> int:
        """
        Get count for a specific n-gram pattern.

        Args:
            rule (List[Optional[int]]): List of token IDs (must be uint16). Example: [None, 1, 2]
        """
        ...

    def fit_smoothing(self, smoothing_name: Optional[str] = None) -> None:
        """
        Fit the smoothing model with the given smoothing name.

        Args:
            smoothing_name (Optional[str]): Smoothing method name (default: 'modified_kneser_ney')
        """
        ...

    def debug_cache_sizes(self) -> None:
        """
        Writes cache sizes to the log (Debug level).
        """
        ...

    def get_smoothed_probabilities(self, history: List[int], rule_set: Optional[List[str]] = None) -> List[Tuple[str, List[Tuple[int, float]]]]:
        """
        Get smoothed probabilities for a given history.

        Args:
            history (List[int]): List of token IDs (must be uint16)
            rule_set (Optional[List[str]]): List of smoothing rules. Example: ['+', '++', '+*-']
        """
        ...

    def get_unsmoothed_probabilities(self, history: List[int]) -> List[Tuple[str, List[Tuple[int, float]]]]:
        """
        Get unsmoothed probabilities for a given history.

        Args:
            history (List[int]): List of token IDs (must be uint16)
        """
        ...

    def probability_for_token(self, history: List[int], predict: int) -> List[Tuple[str, float]]:
        """
        Get probability distribution for a specific token.

        Args:
            history (List[int]): List of token IDs (must be uint16)
            predict (int): Token ID to predict
        """
        ...
