use crate::trienode::TrieNode;
use crate::smoothing::Smoothing;
use serde::{Serialize, Deserialize};
use std::mem;
use std::fs::{File, metadata};
use std::io::{BufReader, BufWriter};
use std::time::Instant;
use std::sync::Arc;
use std::ops::Range;
use bincode::{serialize_into, deserialize_from};
use tqdm::tqdm;
use hashbrown::{HashMap, HashSet};

#[derive(Serialize, Deserialize, Clone)]
pub struct NGramTrie {
    pub root: Box<TrieNode>,
    pub n_gram_max_length: u32,
    pub rule_set: Vec<String>
}

impl NGramTrie {
    pub fn new(n_gram_max_length: u32) -> Self {
        let _rule_set = NGramTrie::_calculate_ruleset(n_gram_max_length - 1);
        NGramTrie {
            root: Box::new(TrieNode::new()),
            n_gram_max_length,
            rule_set: _rule_set
        }
    }

    //better to use this as it is simle, maybe even faster
    pub fn insert_recursive(&mut self, n_gram: &[u16]) {
        self.root.insert_recursive(n_gram);
    }
    
    #[deprecated]
    pub fn insert(&mut self, n_gram: &[u16]) {
        let mut current_node = &mut self.root;
        current_node.count += 1;
        for i in 0..n_gram.len() {
            current_node = current_node.children.entry(n_gram[i]).or_insert_with(|| Box::new(TrieNode::new()));
            current_node.count += 1;
        }
    }

    //better to use this as it is simle
    pub fn merge_recursive(&mut self, other: &NGramTrie) {
        self.root.merge_recursive(&other.root);
    }

    #[deprecated] //cant really work
    pub fn merge_shit(&mut self, other: &NGramTrie) {
        let mut stack = vec![(self.root.as_mut() as *mut TrieNode, other.root.as_ref() as *const TrieNode)];

        unsafe {
            while let Some((self_node_ptr, other_node_ptr)) = stack.pop() {
                let self_node = &mut *self_node_ptr;
                let other_node = &*other_node_ptr;

                for (key, other_child) in &other_node.children {
                    let self_child = self_node.children.entry(*key).or_insert_with(|| Box::new(TrieNode::new()));
                    self_child.count += other_child.count;
                    stack.push((self_child.as_mut() as *mut TrieNode, other_child.as_ref() as *const TrieNode));
                }
            }
        }
    }

    //better to use size_in_ram, faster by 7-10%
    pub fn size_in_ram_recursive(&self) -> usize {
        mem::size_of::<NGramTrie>() + self.root.size_in_ram_recursive()
    }

    #[deprecated]
    pub fn size_in_ram(&self) -> usize {
        let mut total_size = mem::size_of::<NGramTrie>();
        let mut stack = vec![self.root.as_ref()];
        while let Some(node) = stack.pop() {
            total_size += mem::size_of::<TrieNode>() + node.children.capacity() * mem::size_of::<(u16, Box<TrieNode>)>();
            for child in node.children.values() {
                stack.push(child);
            }
        }
        total_size
    }

    pub fn save(&self, filename: &str) -> std::io::Result<()> {
        println!("----- Saving trie -----");
        let start = Instant::now();
        let file = File::create(filename)?;
        let writer = BufWriter::new(file);
        serialize_into(writer, self).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let duration = start.elapsed();
        println!("Time taken to save trie: {:?}", duration);
        let file_size = metadata(filename).expect("Unable to get file metadata").len();
        let file_size_mb = file_size as f64 / (1024.0 * 1024.0);
        println!("Size of saved file: {:.2} MB", file_size_mb);
        Ok(())
    }

    pub fn load(filename: &str) -> std::io::Result<Self> {
        println!("----- Loading trie -----");
        let start = Instant::now();
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        let trie: NGramTrie = deserialize_from(reader).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let duration = start.elapsed();
        println!("Time taken to load trie: {:?}", duration);
        println!("Size of loaded trie in RAM: {} MB", trie.size_in_ram_recursive() as f64 / (1024.0 * 1024.0));
        Ok(trie)
    }

    pub fn _preprocess_rule_context(tokens: &[u16], rule_context: Option<&str>) -> Vec<Option<u16>> {
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

    pub fn _calculate_ruleset(n_gram_max_length: u32) -> Vec<String> {
        if n_gram_max_length == 1 {
            return vec!["+".to_string(), "-".to_string()];
        }
        let mut ruleset = Vec::<String>::new();
        ruleset.extend(NGramTrie::_calculate_ruleset(n_gram_max_length - 1));
    
        let characters = vec!["+", "*", "-"];
        
        let mut combinations : Vec<String> = (2..n_gram_max_length).fold(
            characters.iter().map(|c| characters.iter().map(move |&d| d.to_owned() + *c)).flatten().collect(),
            |acc,_| acc.into_iter().map(|c| characters.iter().map(move |&d| d.to_owned() + &*c)).flatten().collect()
        );
    
        combinations.retain(|comb| comb.starts_with('+'));
    
        let mut tokens = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789".to_string();
        tokens.truncate(n_gram_max_length as usize);
        let mut hashmap = HashMap::<String, String>::new();
    
        for comb in combinations {
            let mut key = "".to_string();
            for (token, rule) in tokens.chars().zip(comb.chars()) {
                match rule {
                    '*' => key += "*",
                    '-' => continue,
                    _ => key += &token.to_string(),
                }
            }
            hashmap.insert(key, comb);
        }
    
        ruleset.extend(hashmap.values().cloned());
    
        ruleset
    }

    pub fn set_rule_set(&mut self, rule_set: Vec<String>) {
        println!("----- Setting rule set -----");
        self.rule_set = rule_set;
        println!("Rule set: {:?}", self.rule_set);
    }

    pub fn get_count(&self, rule: &[Option<u16>]) -> u32 {
        self.root.get_count(rule)
    }

    //TODO: merge with unique_continuation_count?
    pub fn find_all_nodes(&self, rule: &[Option<u16>]) -> Vec<&TrieNode> {
        self.root.find_all_nodes(rule)
    }

    pub fn unique_continuations(&self, rule: &[Option<u16>]) -> HashSet<u16> {
        let mut unique = HashSet::<u16>::new();
        for node in self.find_all_nodes(rule) {
            unique.extend(node.children.keys());
        }
        unique
    }

    //TODO: Cache??
    pub fn probability_for_token(&self, smoothing: &impl Smoothing, history: &[u16], predict: u16) -> Vec<(String, f64)> {
        let mut rules_smoothed = Vec::<(String, f64)>::new();

        for r_set in &self.rule_set.iter().filter(|r| r.len() <= history.len()).collect::<Vec<_>>()[..] {
            let mut rule = NGramTrie::_preprocess_rule_context(history, Some(&r_set));
            rule.push(Some(predict));
            rules_smoothed.push((r_set.to_string(), smoothing.smoothing(&self, &rule)));
        }

        rules_smoothed
    }

    pub fn get_prediction_probabilities(&self, smoothing: &impl Smoothing, history: &[u16]) -> Vec<(u16, Vec<(String, f64)>)> { 
        let mut prediction_probabilities = Vec::<(u16, Vec<(String, f64)>)>::new();

        for token in self.root.children.keys() {
            let probabilities = self.probability_for_token(smoothing, history, *token);
            prediction_probabilities.push((*token, probabilities));
        }

        prediction_probabilities
    }

    pub fn fit(tokens: Arc<Vec<u16>>, n_gram_max_length: u32, max_tokens: Option<usize>) -> Self {
        println!("----- Trie fitting -----");
        let x = tokens.len() as f64; //450_000_000;
        let y = 0.0017 * x.powf(0.8814);
        let _x = (y / 0.0017).powf(1.0 / 0.8814) as f64;
        let t = (0.000003 * x - 0.533) / 60.0;
        println!("Expected time for {} tokens: {} min", x, t);
        println!("Expected ram usage for {} tokens: {} MB", x, y);
        let start = Instant::now();
        let mut trie = NGramTrie::new(n_gram_max_length);
        let max_tokens = max_tokens.unwrap_or(tokens.len()).min(tokens.len());
        for i in tqdm(0..max_tokens - n_gram_max_length as usize + 1) {
            trie.insert_recursive(&tokens[i..i + n_gram_max_length as usize]);
        }
        let duration = start.elapsed();
        println!("Time taken to fit trie: {:?}", duration);
        println!("Size of trie in RAM: {} MB", trie.size_in_ram_recursive() as f64 / (1024.0 * 1024.0));
        trie
    }

    pub fn fit_multithreaded(tokens: Arc<Vec<u16>>, ranges: Vec<Range<usize>>, n_gram_max_length: u32) -> Self {
        let mut trie = NGramTrie::new(n_gram_max_length);

        let mut handles = vec![];

        for range in ranges {
            let mut trie_clone = trie.clone();

            let _tokens = tokens.clone();

            let handle = std::thread::spawn(move || {
                for i in range.start..range.end - n_gram_max_length as usize + 1 {
                    let n_gram = &_tokens[i..i + n_gram_max_length as usize];
                    trie_clone.insert_recursive(n_gram);
                }
                trie_clone
            });

            handles.push(handle);
        }

        for handle in handles {
            let partial_trie = handle.join().unwrap();
            trie.merge_recursive(&partial_trie);
        }
        trie
    }

    pub fn fit_multithreaded_recursively(tokens: Arc<Vec<u16>>, ranges: Vec<Range<usize>>, n_gram_max_length: u32) -> Self {
        if ranges.len() > 1 {
            let mid = ranges.len() / 2;
            let left = ranges[..mid].to_vec();
            let right = ranges[mid..].to_vec();
            // Recursively process both halves
            let right_clone = tokens.clone();
            let handle = std::thread::spawn(move || {
                NGramTrie::fit_multithreaded_recursively(right_clone, right, n_gram_max_length)
            });
            let mut left_trie = NGramTrie::fit_multithreaded_recursively(tokens, left, n_gram_max_length);
            let right_trie = handle.join().unwrap();
            left_trie.merge_recursive(&right_trie);
            left_trie
        } else {
            let mut trie = NGramTrie::new(n_gram_max_length);
            let range = &ranges[0];
            for i in range.start..range.end - n_gram_max_length as usize + 1 {
                let n_gram = &tokens[i..i + n_gram_max_length as usize];
                trie.insert_recursive(n_gram);
            }
            trie
        }
    }

    pub fn load_json(filename: &str, max_tokens: Option<usize>) -> std::io::Result<Arc<Vec<u16>>> {
        println!("----- Loading tokens -----");
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        let start = std::time::Instant::now();
        let mut tokens: Vec<u16> = serde_json::from_reader(reader).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let duration = start.elapsed();
        println!("Time taken to load tokens: {:?}", duration);
        println!("Size of tokens in RAM: {:.2} MB", (tokens.len() * std::mem::size_of::<u16>()) as f64 / 1024.0 / 1024.0);
        if let Some(max) = max_tokens {
            if max < tokens.len() {
                tokens.truncate(max);
            }
        }
        println!("Size of tokens in RAM after truncation: {:.2} MB", (tokens.len() * std::mem::size_of::<u16>()) as f64 / 1024.0 / 1024.0);
        println!("Tokens loaded: {}", tokens.len());
        Ok(Arc::new(tokens))
    }
    
    pub fn split_into_ranges(tokens: Arc<Vec<u16>>, max_tokens: usize, number_of_chunks: usize, n_gram_max_length: u32) -> Vec<Range<usize>> {
        let mut ranges = Vec::new();
        let max_tokens = std::cmp::min(max_tokens, tokens.len());
        let chunk_size = (max_tokens as f64 / number_of_chunks as f64).ceil() as usize;
        for i in 0..number_of_chunks {
            let start = i * chunk_size;
            let end = if i == number_of_chunks - 1 {
                max_tokens
            } else {
                (i + 1) * chunk_size + n_gram_max_length as usize - 1
            };
            ranges.push(start..end);
        }
        ranges
    }

}