#![allow(warnings)]
pub mod trie;
pub mod smoothing;
pub mod smoothed_trie;

use trie::NGramTrie;
use trie::trienode::TrieNode;
use smoothing::ModifiedBackoffKneserNey;
use sorted_vector_map::SortedVectorMap;
use smoothed_trie::SmoothedTrie;

use std::sync::Arc;
use serde::Serialize;
use serde::Deserialize;
use std::time::Instant;
use std::fs::OpenOptions;
use std::io::Write;
use actix_web::{web, App, HttpServer, Responder};

fn test_performance_and_write_stats(tokens: Arc<Vec<u16>>, data_sizes: Vec<usize>, n_gram_lengths: Vec<u32>, output_file: &str) {
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .append(true)
        .open(output_file)
        .unwrap();

    writeln!(file, "Data Size,N-gram Length,Fit Time (s),RAM Usage (MB)").unwrap();

    let num_threads = std::thread::available_parallelism()
        .map(|p| p.get()).unwrap_or(1);

    for data_size in data_sizes {
        for n_gram_length in &n_gram_lengths {
            //let ranges = NGramTrie::split_into_ranges(tokens.clone(), data_size, num_threads, *n_gram_length);
            // Measure fit time
            let start = Instant::now();
            //let trie = NGramTrie::fit_multithreaded(tokens.clone(), ranges, *n_gram_length);
            //let trie = NGramTrie::fit_multithreaded_recursively(tokens.clone(), ranges, *n_gram_length);
            let trie = NGramTrie::fit(tokens.clone(), *n_gram_length, None,Some(data_size));
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
    println!("----- Starting performance tests -----");
    let tokens = NGramTrie::load_json(filename, Some(100_000_000)).unwrap();
    println!("Tokens loaded: {}", tokens.len());
    let data_sizes = (1..10).map(|x| x * 1_000_000).chain((1..=10).map(|x| x * 10_000_000)).collect::<Vec<_>>();
    let n_gram_lengths = [7].to_vec();
    let output_file = "fit_sorted_vector_map_with_box.csv";

    test_performance_and_write_stats(tokens, data_sizes, n_gram_lengths, output_file);
}

#[derive(Serialize, Deserialize)]
struct PredictionRequest {
    history: Vec<u16>,
    predict: u16,
}

#[derive(Serialize)]
struct PredictionResponse {
    probabilities: Vec<(u16, Vec<(String, f64)>)>,
}

async fn predict_probability(req: web::Json<PredictionRequest>, smoothed_trie: web::Data<SmoothedTrie>) -> impl Responder {
    let mut probabilities = smoothed_trie.get_prediction_probabilities(&req.history);

    probabilities.sort_by_key(|k| k.0);

    let response = PredictionResponse {
        probabilities: probabilities,
    };
    web::Json(response)
}

#[tokio::main]
async fn start_http_server(smoothed_trie: Arc<SmoothedTrie>) -> std::io::Result<()> {
    println!("----- Starting HTTP server -----");
    HttpServer::new(move || {
        App::new()
            .app_data(smoothed_trie.clone())
            .service(web::resource("/predict").route(web::post().to(predict_probability)))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}

fn main() {
    //run_performance_tests("tokens.json");

    //NGramTrie::estimate_time_and_ram(475_000_000);

    let mut rule_set = NGramTrie::_calculate_ruleset(5);
    let tokens = NGramTrie::load_json("../170k_tokens.json", None).unwrap();

    let mut smoothed_trie = SmoothedTrie::new(NGramTrie::new(7, Some(2_usize.pow(14))), Box::new(ModifiedBackoffKneserNey::new(&NGramTrie::new(7, Some(2_usize.pow(14))))));

    smoothed_trie.fit(tokens, 7, Some(2_usize.pow(14)), None);
    smoothed_trie.fit_smoothing();

    smoothed_trie.set_rule_set(vec!["++++++".to_string(), "+++++".to_string(), "++++".to_string(), "+++".to_string(), "++".to_string(), "+".to_string()]);

    println!("----- Getting rule count -----"); //4107, 1253, 375, 4230, 1140, 3042 ;;; 510, 224, 290, 185, 1528, 135
    let rule = NGramTrie::_preprocess_rule_context(&vec![4107, 1253, 375, 4230, 1140, 3042], Some("**+***"));
    let start = Instant::now();
    let count = smoothed_trie.get_count(rule);
    let elapsed = start.elapsed();
    println!("Count: {}", count);
    println!("Time taken: {:?}", elapsed);
    
    let probabilities = smoothed_trie.get_prediction_probabilities(&vec![4107, 1253, 375, 4230, 1140, 3042]);
        
}