use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use bincode::deserialize_from;
use std::io::BufWriter;
use bincode::serialize_into;
use serde::Serialize;
use serde::Deserialize;

#[derive(Serialize, Deserialize)]
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
#[derive(Serialize, Deserialize)]
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

    fn save(&self, filename: &str) -> std::io::Result<()> {
        let file = File::create(filename)?;
        let writer = BufWriter::new(file);
        serialize_into(writer, self).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        Ok(())
    }

    fn load(filename: &str) -> std::io::Result<Self> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        let trie: NGramTrie = deserialize_from(reader).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(trie)
    }
    
}





fn main() {
    let mut trie = NGramTrie::new(3);
    trie.insert(&[1, 2, 3]);
    trie.insert(&[1, 2, 4]);
    trie.insert(&[2, 3, 4]);
    trie.insert(&[3, 4, 5]);
    println!("{:?}", trie.search(&[1, 2, 3], Some("++*")));
    println!("{:?}", trie.search(&[2, 3, 4], Some("+++")));
    trie.save("trie.bin").expect("Failed to save trie");
    let loaded_trie = NGramTrie::load("trie.bin").unwrap();
    println!("{:?}", loaded_trie.search(&[1, 2, 3], Some("++*")));
    println!("{:?}", loaded_trie.search(&[2, 3, 4], Some("+++")));
}
