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
            trie.insert_recursive(&tokens[i..i + n_gram_max_length as usize]);
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
    //run_performance_tests(filename);

    let x = 50_000_000;
    let y = 0.0017 * (x as f64).powf(0.8814) / 1024.0;
    let _x = (y / 0.0017).powf(1.0 / 0.8814) as u32;
    println!("Expected ram usage for {} tokens: {} GB", x, y);
}