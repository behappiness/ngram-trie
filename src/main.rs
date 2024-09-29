use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use bincode::deserialize_from;
use std::io::BufWriter;
use bincode::serialize_into;
use serde::Serialize;
use serde::Deserialize;
use serde_json;

#[derive(Serialize, Deserialize, Clone)]
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

#[derive(Serialize, Deserialize, Clone)]
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

    fn fit(tokens: &[u32], n_gram_max_length: u32) -> Self {
        let mut trie = NGramTrie::new(n_gram_max_length);
        for i in 0..tokens.len() - n_gram_max_length as usize + 1 {
            let n_gram = &tokens[i..i + n_gram_max_length as usize];
            trie.insert(n_gram);
        }
        trie
    }

    fn fit_multithreaded(tokens: &[u32], n_gram_max_length: u32) -> Self {
        let mut trie = NGramTrie::new(n_gram_max_length);

        let num_threads = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1);
        println!("Using {} threads for multithreaded fitting", num_threads);
        
        let chunk_size = (tokens.len() - n_gram_max_length as usize + 1) / num_threads;
        let mut handles = vec![];

        for chunk_start in (0..tokens.len() - n_gram_max_length as usize + 1).step_by(chunk_size) {
            let chunk_end = std::cmp::min(chunk_start + chunk_size, tokens.len() - n_gram_max_length as usize + 1);
            let tokens_slice = tokens[chunk_start..].to_vec();
            let mut trie_clone = trie.clone();
            let n_gram_max_length_clone = n_gram_max_length;

            let handle = std::thread::spawn(move || {
                for i in 0..chunk_end - chunk_start {
                    let n_gram = &tokens_slice[i..i + n_gram_max_length_clone as usize];
                    trie_clone.insert(n_gram);
                }
                trie_clone
            });

            handles.push(handle);
        }

        for handle in handles {
            let partial_trie = handle.join().unwrap();
            trie.merge(&partial_trie);
        }
        trie
    }

    /// Merges another trie into this one
    fn merge(&mut self, other: &NGramTrie) {
        NGramTrie::merge_nodes(&mut self.root, &other.root);
    }

    fn merge_nodes(current: &mut TrieNode, other: &TrieNode) {
        // Update the count for the current node
        current.count += other.count;

        // Merge children
        for (char, other_child) in &other.children {
            match current.children.entry(*char) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    // If the current trie doesn't have this child, clone the other's child
                    entry.insert(other_child.clone());
                }
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    // If both tries have this child, recursively merge them
                    NGramTrie::merge_nodes(entry.get_mut(), other_child);
                }
            }
        }
    }

    fn size_in_ram(node: &TrieNode) -> usize {
        let mut size = std::mem::size_of::<TrieNode>();
        size += node.children.capacity() * std::mem::size_of::<(u32, TrieNode)>();
        for child in node.children.values() {
            size += Self::size_in_ram(child);
        }
        size
    }
    
    
}

fn load_tokens_from_json(filename: &str) -> std::io::Result<Vec<u32>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let tokens: Vec<u32> = serde_json::from_reader(reader).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    Ok(tokens)
}



fn main() {
    let tokens = load_tokens_from_json("/home/boti/Desktop/ngram-llm-analysis/data/170k_small_tokenized_data.json").unwrap();
    println!("Loaded {} tokens from JSON", tokens.len());

    //let tokens = vec![1, 2, 3, 1, 2, 4, 2, 3, 4, 3, 4, 5];
    let start = std::time::Instant::now();
    let _trie = NGramTrie::fit(&tokens, 3);
    let duration = start.elapsed();
    println!("Time taken to fit NGramTrie: {:?}", duration);

    let start = std::time::Instant::now();
    let trie = NGramTrie::fit_multithreaded(&tokens, 3);
    let duration = start.elapsed();
    println!("Time taken to fit NGramTrie: {:?}", duration);

    let start = std::time::Instant::now();
    let result = trie.search(&[4038, 2193], Some("++"));
    let duration = start.elapsed();
    println!("Result: {:?}", result);
    println!("Time taken: {:?}", duration);


    println!("{:?}", trie.search(&[4038, 2193, 2332], Some("+++")));

    
    println!("{:?}", trie.search(&[4038, 2193, 2332], Some("++*")));

    trie.save("trie.bin").expect("Failed to save trie");
    let loaded_trie = NGramTrie::load("trie.bin").unwrap();

    println!("{:?}", loaded_trie.search(&[4038, 2193, 2332], Some("++*")));
    
    
    let total_size = NGramTrie::size_in_ram(&trie.root);
    println!("Total size of the trie (including all nodes): {:.2} MB", total_size as f64 / (1024.0 * 1024.0));

}
