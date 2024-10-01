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

        let chunk_size = (tokens.len() as f64 / num_threads as f64).ceil() as usize;

        let mut chunks: Vec<Vec<u32>> = Vec::new();

        for i in 0..num_threads {
            //println!("Processing chunk {}", i);
            let start = i * chunk_size;
            let end = if i == num_threads - 1 {
                tokens.len()
            } else {
                (i + 1) * chunk_size + n_gram_max_length as usize - 1
            };
            chunks.push(tokens[start..end].to_owned());
        }

        for chunk in &mut chunks {
            chunk.shrink_to_fit();
        }
        chunks.shrink_to_fit();

        //println!("Number of chunks: {}", chunks.len());

        let mut handles = vec![];

        for chunk in chunks {
            let mut trie_clone = trie.clone();

            let handle = std::thread::spawn(move || {
                for i in 0..chunk.len() - n_gram_max_length as usize + 1 {
                    let n_gram = &chunk[i..i + n_gram_max_length as usize];
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

    fn multithred_recursively(chunks: Arc<Vec<Vec<u32>>>, range: Range<usize>, n_gram_max_length: u32) -> NGramTrie {
        if range.len() > 1 {
            let mid = (range.start + range.end) / 2;
            let left = range.start..mid;
            let right = mid..range.end;
            // Recursively process both halves
            let right_clone = chunks.clone();
            let handle = std::thread::spawn(move || {
                NGramTrie::multithred_recursively(right_clone, right, n_gram_max_length)
            });
            let mut left_trie = NGramTrie::multithred_recursively(chunks, left, n_gram_max_length);
            let right_trie = handle.join().unwrap();
            left_trie.merge(&right_trie);
            left_trie
        } else {
            let mut trie = NGramTrie::new(n_gram_max_length);
            for i in 0..chunks[range.start].len() - n_gram_max_length as usize + 1 {
                let n_gram = &chunks[range.start][i..i + n_gram_max_length as usize];
                trie.insert(n_gram);
            }
            trie
        }
    }

    fn load_and_split_huge_file(file_path: &str, n_gram_max_length: u32) -> Result<Arc<Vec<Vec<u32>>>, std::io::Error> {
        // Load the file
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        let numbers: Vec<u32> = serde_json::from_reader(reader)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    
        // Get the number of CPU cores
        let num_cores = std::thread::available_parallelism()
            .map(|p| p.get()).unwrap_or(1);
    
        // Calculate chunk size
        let chunk_size = ((numbers.len() as f64) / (num_cores as f64)).ceil() as usize;
    
        // Split into chunks
        let mut chunks: Vec<Vec<u32>> = Vec::new();

        for i in 0..num_cores {
            //println!("Processing chunk {}", i);
            let start = i * chunk_size;
            let end = if i == num_cores - 1 {
                numbers.len()
            } else {
                (i + 1) * chunk_size + n_gram_max_length as usize - 1
            };
            chunks.push(numbers[start..end].to_owned());
        }
    
        // Wrap in Arc and return
        Ok(Arc::new(chunks))
    }

    /// Merges another trie into this one
    fn merge(&mut self, other: &NGramTrie) {
        self.root.merge(&other.root);
    }

    fn size_in_ram(&self) -> usize {
        mem::size_of::<NGramTrie>() + self.root.size_in_ram()
    }
    
    
}

fn load_tokens_from_json(filename: &str) -> std::io::Result<Vec<u32>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let tokens: Vec<u32> = serde_json::from_reader(reader).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    Ok(tokens)
}

fn main() {
    let filename = "/home/boti/Desktop/ngram-llm-analysis/data/170k_small_tokenized_data.json";
    let n_gram_max_length = 30;

    // {   // old approach (single threaded)
    //     println!("Old approach (single threaded):");
    //     let tokens = load_tokens_from_json(filename).unwrap();
    //     println!("Loaded {} tokens from JSON", tokens.len());

    //     let start = std::time::Instant::now();
    //     let trie = NGramTrie::fit(&tokens, n_gram_max_length);
    //     let duration = start.elapsed();
    //     println!("Time taken to fit NGramTrie: {:?}", duration);

    //     let start = std::time::Instant::now();
    //     let result1 = trie.search(&[4038, 2193, 2332], Some("+++"));
    //     let duration1 = start.elapsed();
    //     println!("Result for '+++': {:?}", result1);
    //     println!("Time taken for '+++': {:?}", duration1);

    //     let start = std::time::Instant::now();
    //     let result2 = trie.search(&[4038, 2193, 2332], Some("+*+"));
    //     let duration2 = start.elapsed();
    //     println!("Result for '+*+': {:?}", result2);
    //     println!("Time taken for '+*+': {:?}", duration2);

    //     let total_size = trie.size_in_ram();
    //     println!("Total size of the trie (including all nodes): {:.2} MB", total_size as f64 / (1024.0 * 1024.0));
    // }

    {   // old approach (multithreaded)
        println!("Old approach (multithreaded):");
        let tokens = load_tokens_from_json(filename).unwrap();
        println!("Loaded {} tokens from JSON", tokens.len());

        let start = std::time::Instant::now();
        let trie = NGramTrie::fit_multithreaded(&tokens, n_gram_max_length);
        let duration = start.elapsed();
        println!("Time taken to fit NGramTrie: {:?}", duration);

        println!("Number of children: {}", trie.root.children.capacity());

        let start = std::time::Instant::now();
        let result1 = trie.search(&[4038, 2193, 2332], Some("+++"));
        let duration1 = start.elapsed();
        println!("Result for '+++': {:?}", result1);
        println!("Time taken for '+++': {:?}", duration1);

        let start = std::time::Instant::now();
        let result2 = trie.search(&[4038, 2193, 2332], Some("++*"));
        let duration2 = start.elapsed();
        println!("Result for '++*': {:?}", result2);
        println!("Time taken for '++*': {:?}", duration2);

        let total_size = trie.size_in_ram();
        println!("Total size of the trie (including all nodes): {:.2} MB", total_size as f64 / (1024.0 * 1024.0));
    }

    {   // new approach (multithreaded and recursively) 
        println!("New approach (multithreaded and recursively):");
        let chunks = NGramTrie::load_and_split_huge_file(filename, n_gram_max_length).unwrap();
        let chunks_len = chunks.len();
        println!("Loaded {} chunks from JSON", chunks_len);
    
        let start = std::time::Instant::now();
        let trie = NGramTrie::multithred_recursively(chunks, 0..chunks_len, n_gram_max_length);
        let duration = start.elapsed();
        println!("Time taken to fit NGramTrie: {:?}", duration);

        let start = std::time::Instant::now();
        let result1 = trie.search(&[4038, 2193, 2332], Some("+++"));
        let duration1 = start.elapsed();
        println!("Result for '+++': {:?}", result1);
        println!("Time taken for '+++': {:?}", duration1);

        let start = std::time::Instant::now();
        let result2 = trie.search(&[4038, 2193, 2332], Some("++*"));
        let duration2 = start.elapsed();
        println!("Result for '++*': {:?}", result2);
        println!("Time taken for '++*': {:?}", duration2);

        let total_size = trie.size_in_ram();
        println!("Total size of the trie (including all nodes): {:.2} MB", total_size as f64 / (1024.0 * 1024.0));
    }
}
