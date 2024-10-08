use std::sync::Arc;
use std::ops::Range;
use std::fs::File;
use std::io::BufReader;
use bincode::deserialize_from;
use hashbrown::HashMap;
use std::io::BufWriter;
use bincode::serialize_into;
use serde::Serialize;
use serde::Deserialize;
use serde_json;
use std::mem;
use fxhash::FxHashMap;
use std::time::Instant;
use std::fs::OpenOptions;
use std::io::Write;
use std::collections::HashSet;
use actix_web::{web, App, HttpServer, Responder};
use tqdm::tqdm;
use std::fs::metadata;

trait Smoothing: Clone{
    fn smoothing(&self, trie: &NGramTrie, rule: &[Option<u32>]) -> f64;
}

#[derive(Clone)]
struct ModifiedBackoffKneserNey {
    d1: f64,
    d2: f64,
    d3: f64,
    uniform: f64
}

impl ModifiedBackoffKneserNey {
    fn new(trie: &NGramTrie) -> Self {
        let (_d1, _d2, _d3) = Self::calculate_d_values(trie);
        let uniform = 1.0 / trie.root.children.len() as f64;
        ModifiedBackoffKneserNey {
            d1: _d1,
            d2: _d2,
            d3: _d3,
            uniform: uniform
        }
    }

    fn calculate_d_values(trie: &NGramTrie) -> (f64, f64, f64) {
        let mut n1: u32 = 0;
        let mut n2: u32 = 0;
        let mut n3: u32 = 0;
        let mut n4: u32 = 0;
        for i in 1..=trie.n_gram_max_length {
            let rule: Vec<Option<u32>> = vec![None; i as usize];
            for node in trie.find_all_nodes(&rule) {
                match node.count {
                    1 => n1 += 1,
                    2 => n2 += 1,
                    3 => n3 += 1,
                    4 => n4 += 1,
                    _ => ()
                }
            }
        }

        if n1 == 0 || n2 == 0 || n3 == 0 || n4 == 0 {
            return (0.1, 0.2, 0.3);
        }

        let y = n1 as f64 / (n1 + 2 * n2) as f64;
        let d1 = 1.0 - 2.0 * y * (n2 as f64 / n1 as f64);
        let d2 = 2.0 - 3.0 * y * (n3 as f64 / n2 as f64);
        let d3 = 3.0 - 4.0 * y * (n4 as f64 / n3 as f64);
        (d1, d2, d3)
    }

    //TODO: Cache
    fn count_unique_ns(trie: &NGramTrie, rule: &[Option<u32>]) -> (u32, u32, u32) {
        let mut n1 = HashSet::<u32>::new();
        let mut n2 = HashSet::<u32>::new();
        let mut n3 = HashSet::<u32>::new();
        for node in trie.find_all_nodes(&rule) {
            for (key, child) in &node.children {
                match child.count {
                    1 => { n1.insert(*key); },
                    2 => { n2.insert(*key); },
                    _ => { n3.insert(*key); }
                }
            }
        }
        (n1.len() as u32, n2.len() as u32, n3.len() as u32)
    }
}

//From Chen & Goodman 1998
impl Smoothing for ModifiedBackoffKneserNey {
    //TODO: Cache
    fn smoothing(&self, trie: &NGramTrie, rule: &[Option<u32>]) -> f64 {
        if rule.len() <= 0 {
            return self.uniform;
        }

        let W_i = &rule[rule.len() - 1];
        let W_i_minus_1 = &rule[..rule.len() - 1];

        let C_i = trie.get_count(&rule);
        let C_i_minus_1 = trie.get_count(&W_i_minus_1);

        let d = match C_i {
            0 => 0.0,
            1 => self.d1,
            2 => self.d2,
            _ => self.d3
        };

        let (n1, n2, n3) = ModifiedBackoffKneserNey::count_unique_ns(trie, &W_i_minus_1);

        let gamma = (self.d1 * n1 as f64 + self.d2 * n2 as f64 + self.d3 * n3 as f64) / C_i_minus_1 as f64;

        return (C_i as f64 - d).max(0.0) / C_i_minus_1 as f64 + gamma * self.smoothing(trie, &rule[1..]);
    }
}

#[derive(Serialize, Deserialize, Clone)]
struct TrieNode {
    children: FxHashMap<u32, Box<TrieNode>>, // maybe u16 is enough
    count: u32
}

impl TrieNode {
    fn new() -> Self {
        TrieNode {
            children: FxHashMap::default(),
            count: 0,
        }
    }

    fn merge_recursive(&mut self, other: &TrieNode) {
        self.count += other.count;
        for (char, other_child) in &other.children {
            self.children
                .entry(*char)
                .or_insert_with(|| Box::new(TrieNode::new()))
                .merge_recursive(other_child);
        }
    }

    fn insert_recursive(&mut self, n_gram: &[u32]) {
        self.count += 1;
        if n_gram.len() == 0 { return; }
        self.children
            .entry(n_gram[0])
            .or_insert_with(|| Box::new(TrieNode::new()))
            .insert_recursive(&n_gram[1..]);
    }

    fn size_in_ram_recursive(&self) -> usize {
        let mut size = mem::size_of::<TrieNode>();
        size += self.children.capacity() * mem::size_of::<(u32, Box<TrieNode>)>();
        for child in self.children.values() {
            size += child.size_in_ram_recursive();
        }
        size
    }

    fn find_all_nodes(&self, rule: &[Option<u32>]) -> Vec<&TrieNode> {
        if rule.len() == 0 { return vec![self]; }
        else {
            let mut nodes = Vec::<&TrieNode>::new();
            match rule[0] {
                None => {
                    for child_node in self.children.values() {
                        nodes.extend(child_node.find_all_nodes(&rule[1..]));
                    }
                },
                Some(token) => {
                    if let Some(child_node) = self.children.get(&token) {
                        nodes.extend(child_node.find_all_nodes(&rule[1..]));
                    }
                }
            }
            nodes
        }
    }
    
    fn get_count(&self, rule: &[Option<u32>]) -> u32 {
        if rule.len() == 0 { return self.count; }
        else {
            match rule[0] {
                None => self.children.values()
                    .map(|child| child.get_count(&rule[1..]))
                    .sum(),
                Some(token) => self.children.get(&token)
                    .map_or(0, |child| child.get_count(&rule[1..]))
            }
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
struct NGramTrie {
    root: Box<TrieNode>,
    n_gram_max_length: u32,
    rule_set: Vec<String>
}

impl NGramTrie {
    fn new(n_gram_max_length: u32) -> Self {
        let _rule_set = NGramTrie::_calculate_ruleset(n_gram_max_length - 1);
        NGramTrie {
            root: Box::new(TrieNode::new()),
            n_gram_max_length,
            rule_set: _rule_set
        }
    }

    //better to use this as it is simle, maybe even faster
    fn insert_recursive(&mut self, n_gram: &[u32]) {
        self.root.insert_recursive(n_gram);
    }
    
    #[deprecated]
    fn insert(&mut self, n_gram: &[u32]) {
        let mut current_node = &mut self.root;
        current_node.count += 1;
        for i in 0..n_gram.len() {
            current_node = current_node.children.entry(n_gram[i]).or_insert_with(|| Box::new(TrieNode::new()));
            current_node.count += 1;
        }
    }

    //better to use this as it is simle
    fn merge_recursive(&mut self, other: &NGramTrie) {
        self.root.merge_recursive(&other.root);
    }

    #[deprecated] //cant really work
    fn merge_shit(&mut self, other: &NGramTrie) {
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
    fn size_in_ram_recursive(&self) -> usize {
        mem::size_of::<NGramTrie>() + self.root.size_in_ram_recursive()
    }

    #[deprecated]
    fn size_in_ram(&self) -> usize {
        let mut total_size = mem::size_of::<NGramTrie>();
        let mut stack = vec![self.root.as_ref()];
        while let Some(node) = stack.pop() {
            total_size += mem::size_of::<TrieNode>() + node.children.capacity() * mem::size_of::<(u32, Box<TrieNode>)>();
            for child in node.children.values() {
                stack.push(child);
            }
        }
        total_size
    }

    fn save(&self, filename: &str) -> std::io::Result<()> {
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

    fn load(filename: &str) -> std::io::Result<Self> {
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

    fn _preprocess_rule_context(tokens: &[u32], rule_context: Option<&str>) -> Vec<Option<u32>> {
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

    fn _calculate_ruleset(n_gram_max_length: u32) -> Vec<String> {
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

    fn set_rule_set(&mut self, rule_set: Vec<String>) {
        println!("----- Setting rule set -----");
        self.rule_set = rule_set;
        println!("Rule set: {:?}", self.rule_set);
    }

    fn get_count(&self, rule: &[Option<u32>]) -> u32 {
        self.root.get_count(rule)
    }

    //TODO: merge with unique_continuation_count?
    fn find_all_nodes(&self, rule: &[Option<u32>]) -> Vec<&TrieNode> {
        self.root.find_all_nodes(rule)
    }

    fn unique_continuations(&self, rule: &[Option<u32>]) -> HashSet<u32> {
        let mut unique = HashSet::<u32>::new();
        for node in self.find_all_nodes(rule) {
            unique.extend(node.children.keys());
        }
        unique
    }

    //TODO: Cache??
    fn probability_for_token(&self, smoothing: &impl Smoothing, history: &[u32], predict: u32) -> Vec<(String, f64)> {
        let mut rules_smoothed = Vec::<(String, f64)>::new();

        for r_set in &self.rule_set.iter().filter(|r| r.len() <= history.len()).collect::<Vec<_>>()[..] {
            let mut rule = NGramTrie::_preprocess_rule_context(history, Some(&r_set));
            rule.push(Some(predict));
            rules_smoothed.push((r_set.to_string(), smoothing.smoothing(&self, &rule)));
        }

        rules_smoothed
    }

    fn get_prediction_probabilities(&self, smoothing: &impl Smoothing, history: &[u32]) -> Vec<(u32, Vec<(String, f64)>)> { 
        let mut prediction_probabilities = Vec::<(u32, Vec<(String, f64)>)>::new();

        for token in self.root.children.keys() {
            let probabilities = self.probability_for_token(smoothing, history, *token);
            prediction_probabilities.push((*token, probabilities));
        }

        prediction_probabilities
    }

    fn fit(tokens: Arc<Vec<u32>>, n_gram_max_length: u32, max_tokens: Option<usize>) -> Self {
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

    fn fit_multithreaded(tokens: Arc<Vec<u32>>, ranges: Vec<Range<usize>>, n_gram_max_length: u32) -> Self {
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

    fn fit_multithreaded_recursively(tokens: Arc<Vec<u32>>, ranges: Vec<Range<usize>>, n_gram_max_length: u32) -> Self {
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

    fn load_json(filename: &str, max_tokens: Option<usize>) -> std::io::Result<Arc<Vec<u32>>> {
        println!("----- Loading tokens -----");
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        let start = std::time::Instant::now();
        let mut tokens: Vec<u32> = serde_json::from_reader(reader).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let duration = start.elapsed();
        println!("Time taken to load tokens: {:?}", duration);
        println!("Size of tokens in RAM: {:.2} MB", (tokens.len() * std::mem::size_of::<u32>()) as f64 / 1024.0 / 1024.0);
        if let Some(max) = max_tokens {
            if max < tokens.len() {
                tokens.truncate(max);
            }
        }
        println!("Size of tokens in RAM after truncation: {:.2} MB", (tokens.len() * std::mem::size_of::<u32>()) as f64 / 1024.0 / 1024.0);
        println!("Tokens loaded: {}", tokens.len());
        Ok(Arc::new(tokens))
    }
    
    fn split_into_ranges(tokens: Arc<Vec<u32>>, max_tokens: usize, number_of_chunks: usize, n_gram_max_length: u32) -> Vec<Range<usize>> {
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

fn test_performance_and_write_stats(tokens: Arc<Vec<u32>>, data_sizes: Vec<usize>, n_gram_lengths: Vec<u32>, output_file: &str) {
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .append(true)
        .open(output_file)
        .unwrap();

    writeln!(file, "Data Size,N-gram Length,Fit Time (ms),RAM Usage (MB)").unwrap();

    let num_threads = std::thread::available_parallelism()
        .map(|p| p.get()).unwrap_or(1);

    for data_size in data_sizes {
        for n_gram_length in &n_gram_lengths {
            //let ranges = NGramTrie::split_into_ranges(tokens.clone(), data_size, num_threads, *n_gram_length);
            // Measure fit time
            let start = Instant::now();
            //let trie = NGramTrie::fit_multithreaded(tokens.clone(), ranges, *n_gram_length);
            //let trie = NGramTrie::fit_multithreaded_recursively(tokens.clone(), ranges, *n_gram_length);
            let trie = NGramTrie::fit(tokens.clone(), *n_gram_length, Some(data_size));
            let fit_time = start.elapsed().as_secs_f64(); 
            // Measure RAM usage
            let ram_usage = trie.size_in_ram_recursive() as f64 / (1024.0 * 1024.0);

            // Write statistics to file
            writeln!(
                file,
                "{},{},{},{:.2}",
                data_size, n_gram_length, fit_time, ram_usage
            ).unwrap();

            println!(
                "Completed: Data Size = {}, N-gram Length = {}, Fit Time = {}, RAM Usage = {:.2} MB",
                data_size, n_gram_length, fit_time, ram_usage
            );
        }
    }
}

fn run_performance_tests(filename: &str) {
    let tokens = NGramTrie::load_json(filename, Some(100_000_000)).unwrap();
    println!("Tokens loaded: {}", tokens.len());
    let data_sizes = (1..10).map(|x| x * 1_000_000).chain((1..=10).map(|x| x * 10_000_000)).collect::<Vec<_>>();
    let n_gram_lengths = [3, 4, 5, 6, 7].to_vec();
    let output_file = "fit_performance.csv";

    test_performance_and_write_stats(tokens, data_sizes, n_gram_lengths, output_file);
}

#[derive(Serialize, Deserialize)]
struct PredictionRequest {
    history: Vec<u32>,
    predict: u32,
}

#[derive(Serialize)]
struct PredictionResponse {
    probabilities: Vec<(u32, Vec<(String, f64)>)>,
}

async fn predict_probability(req: web::Json<PredictionRequest>, trie: web::Data<NGramTrie>, smoothing: web::Data<ModifiedBackoffKneserNey>) -> impl Responder {
    let mut probabilities = trie.get_prediction_probabilities(smoothing.as_ref(), &req.history);

    probabilities.sort_by_key(|k| k.0);

    let response = PredictionResponse {
        probabilities: probabilities,
    };
    web::Json(response)
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let tokens = NGramTrie::load_json("/home/boti/Desktop/ngram-llm-analysis/data/170k_small_tokenized_data.json", None).unwrap();

    let mut trie = NGramTrie::fit(tokens, 7, None);
    
    trie.save("trie.bin");

    trie.set_rule_set(vec!["++++++".to_string()]);

    let smoothing = ModifiedBackoffKneserNey::new(&trie);
    println!("Smoothing calculated, d1: {}, d2: {}, d3: {}, uniform: {}", smoothing.d1, smoothing.d2, smoothing.d3, smoothing.uniform);

    let trie = Arc::new(trie);
    let smoothing = Arc::new(smoothing);

    println!("----- Starting HTTP server -----");
    HttpServer::new(move || {
        App::new()
            .app_data(trie.clone())
            .app_data(smoothing.clone())
            .service(web::resource("/predict").route(web::post().to(predict_probability)))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}