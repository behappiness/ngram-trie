use std::collections::HashMap;

struct TrieNode {
    children: HashMap<u32, TrieNode>, // maybe u16 is enough
    count: u32
}

impl TrieNode {
    fn new() -> Self {
        TrieNode {
            children: HashMap::new(),
            count: 0,
        }
    }
}

// #[derive(Serialize, Deserialize, Debug)]
struct NGramTrie {
    root: TrieNode,
    n_gram_max_length: u32,
}

impl NGramTrie {
    fn new(n_gram_max_length: u32) -> Self {
        NGramTrie {
            root: TrieNode::new(),
            n_gram_max_length,
        }
    }

    fn insert(&mut self, n_gram: &[u32]) {
        assert!(n_gram.len() == self.n_gram_max_length as usize, "N-gram length must be equal to the maximum length");
        let mut node = &mut self.root;

        for &token in n_gram {
            node = node.children.entry(token).or_insert_with(TrieNode::new);
            node.count += 1;
        }
    }

    fn _preprocess_rule_context(&self, tokens: &[u32], rule_context: Option<&str>) -> Vec<Option<u32>> {
        let mut result = Vec::new();
        
        if let Some(rule_context) = rule_context {
            assert_eq!(tokens.len(), rule_context.len(), "Tokens and rule context must be of the same length");
            
            for (&token, rule) in tokens.iter().zip(rule_context.chars()) {
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

    fn search(&self, tokens: &[u32], rule_context: Option<&str>) -> u32 {
        fn _search(node: &TrieNode, rule_context: &[Option<u32>], depth: usize) -> u32 {
            if depth == rule_context.len() { return node.count; }

            let mut total_count = 0;
            let token = rule_context[depth];

            match token {
                None => {
                    for child_node in node.children.values() {
                        total_count += _search(child_node, rule_context, depth + 1);
                    }
                },
                Some(token) => {
                    if let Some(child_node) = node.children.get(&token) {
                        total_count += _search(child_node, rule_context, depth + 1);
                    } else {
                        return 0;
                    }
                }
            }
            
            total_count
        }

        let search_context = self._preprocess_rule_context(tokens, rule_context);
        assert!(search_context.len() <= self.n_gram_max_length as usize, "Search context length must be less than or equal to the maximum length");
        _search(&self.root, &search_context, 0)
    }
}


fn main() {
    let mut trie = NGramTrie::new(3);
    trie.insert(&[1, 2, 3]);
    trie.insert(&[1, 2, 4]);
    trie.insert(&[2, 3, 4]);
    trie.insert(&[3, 4, 5]);
    println!("{:?}", trie.search(&[1, 2, 3], Some("++*")));
}
