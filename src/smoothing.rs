use crate::trie::NGramTrie;
use hashbrown::HashSet;
use tqdm::tqdm;
use std::time::Instant;

pub trait Smoothing: Clone{
    fn smoothing(&self, trie: &NGramTrie, rule: &[Option<u16>]) -> f64;
}

#[derive(Clone)]
pub struct ModifiedBackoffKneserNey {
    pub d1: f64,
    pub d2: f64,
    pub d3: f64,
    pub uniform: f64
}

impl ModifiedBackoffKneserNey {
    pub fn new(trie: &NGramTrie) -> Self {
        let (_d1, _d2, _d3, _uniform) = Self::calculate_d_values(trie);
        ModifiedBackoffKneserNey {
            d1: _d1,
            d2: _d2,
            d3: _d3,
            uniform: _uniform
        }
    }

    pub fn calculate_d_values(trie: &NGramTrie) -> (f64, f64, f64, f64) {
        println!("----- Calculating d values -----");
        let start = Instant::now();
        let mut n1: u32 = 0;
        let mut n2: u32 = 0;
        let mut n3: u32 = 0;
        let mut n4: u32 = 0;
        for i in tqdm(1..=trie.n_gram_max_length) {
            let rule: Vec<Option<u16>> = vec![None; i as usize];
            for node in tqdm(trie.find_all_nodes(&rule)) {
                match node.count {
                    1 => n1 += 1,
                    2 => n2 += 1,
                    3 => n3 += 1,
                    4 => n4 += 1,
                    _ => ()
                }
            }
        }

        let uniform = 1.0 / trie.root.children.len() as f64;

        if n1 == 0 || n2 == 0 || n3 == 0 || n4 == 0 {
            return (0.1, 0.2, 0.3, uniform);
        }

        let y = n1 as f64 / (n1 + 2 * n2) as f64;
        let d1 = 1.0 - 2.0 * y * (n2 as f64 / n1 as f64);
        let d2 = 2.0 - 3.0 * y * (n3 as f64 / n2 as f64);
        let d3 = 3.0 - 4.0 * y * (n4 as f64 / n3 as f64);
        let elapsed = start.elapsed();
        println!("Time taken: {:?}", elapsed);
        println!("Smoothing calculated, d1: {}, d2: {}, d3: {}, uniform: {}", d1, d2, d3, uniform);
        (d1, d2, d3, uniform)
    }

    //TODO: Cache
    pub fn count_unique_ns(trie: &NGramTrie, rule: &[Option<u16>]) -> (u32, u32, u32) {
        let mut n1 = HashSet::<u16>::new();
        let mut n2 = HashSet::<u16>::new();
        let mut n3 = HashSet::<u16>::new();
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
    fn smoothing(&self, trie: &NGramTrie, rule: &[Option<u16>]) -> f64 {
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
