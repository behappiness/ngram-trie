use std::sync::Arc;
use std::ops::Range;
use std::fs::File;
use std::io::BufReader;
use bincode::deserialize_from;
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

    fn merge(&mut self, other: &TrieNode) {
        // Update the count for the current node
        self.count += other.count;

        // Merge children
        for (char, other_child) in &other.children {
            self.children
                .entry(*char)
                .or_insert_with(|| Box::new(TrieNode::new()))
                .merge(other_child);
        }
    }

    fn insert(&mut self, n_gram: &[u32]) {
        self.count += 1;
        if n_gram.len() == 0 { return; }
        self.children
            .entry(n_gram[0])
            .or_insert_with(|| Box::new(TrieNode::new()))
            .insert(&n_gram[1..]);
    }

    fn size_in_ram(&self) -> usize {
        let mut size = mem::size_of::<TrieNode>();
        size += self.children.capacity() * mem::size_of::<(u32, Box<TrieNode>)>();
        for child in self.children.values() {
            size += child.size_in_ram();
        }
        size
    }
}

#[derive(Serialize, Deserialize, Clone)]
struct NGramTrie {
    root: Box<TrieNode>,
    n_gram_max_length: u32,
}

impl NGramTrie {
    fn new(n_gram_max_length: u32) -> Self {
        NGramTrie {
            root: Box::new(TrieNode::new()),
            n_gram_max_length,
        }
    }

    fn insert(&mut self, n_gram: &[u32]) {
        //assert!(n_gram.len() == self.n_gram_max_length as usize, "N-gram length must be equal to the maximum length");
        self.root.insert(n_gram);
    }

    fn merge(&mut self, other: &NGramTrie) {
        self.root.merge(&other.root);
    }

    fn size_in_ram(&self) -> usize {
        mem::size_of::<NGramTrie>() + self.root.size_in_ram()
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

    fn fit(tokens: Arc<Vec<u32>>, n_gram_max_length: u32, max_tokens: Option<usize>) -> Self {
        let mut trie = NGramTrie::new(n_gram_max_length);
        let max_tokens = max_tokens.unwrap_or(tokens.len()).min(tokens.len());
        for i in 0..max_tokens - n_gram_max_length as usize + 1 {
            let n_gram = &tokens[i..i + n_gram_max_length as usize];
            trie.insert(n_gram);
        }
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
            left_trie.merge(&right_trie);
            left_trie
        } else {
            let mut trie = NGramTrie::new(n_gram_max_length);
            let range = &ranges[0];
            for i in range.start..range.end - n_gram_max_length as usize + 1 {
                let n_gram = &tokens[i..i + n_gram_max_length as usize];
                trie.insert(n_gram);
            }
            trie
        }
    }

    fn load_json(filename: &str, max_tokens: Option<usize>) -> std::io::Result<Arc<Vec<u32>>> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        let start = std::time::Instant::now();
        let mut tokens: Vec<u32> = serde_json::from_reader(reader).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let duration = start.elapsed();
        println!("Time taken to load tokens: {:?}", duration);
        println!("Size of tokens in RAM: {} bytes", tokens.len() * std::mem::size_of::<u32>());
        if let Some(max) = max_tokens {
            if max < tokens.len() {
                tokens.truncate(max);
            }
        }
        println!("Size of tokens in RAM after truncation: {} bytes", tokens.len() * std::mem::size_of::<u32>());
        Ok(Arc::new(tokens))
    }
    
    fn split_into_ranges(tokens: Arc<Vec<u32>>, max_tokens: usize, num_threads: usize, n_gram_max_length: u32) -> Vec<Range<usize>> {
        let mut ranges = Vec::new();
        let max_tokens = std::cmp::min(max_tokens, tokens.len());
        let chunk_size = (max_tokens as f64 / num_threads as f64).ceil() as usize;
        for i in 0..num_threads {
            let start = i * chunk_size;
            let end = if i == num_threads - 1 {
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
            let ram_usage = trie.size_in_ram() as f64 / (1024.0 * 1024.0);

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

fn main() {
    let filename = "/home/boti/Desktop/ngram-llm-analysis/data/cleaned_tokenized_data.json";
    run_performance_tests(filename);

    let x = 371710322;
    let y = 0.0015 * (x as f64).powf(0.8891) / 1024.0;
    println!("{}", y);
}
