#![allow(warnings)]
mod trie;

use trie::NGramTrie;
use trie::trienode::TrieNode;
use trie::smoothing::ModifiedBackoffKneserNey;
use sorted_vector_map::SortedVectorMap;

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

async fn predict_probability(req: web::Json<PredictionRequest>, trie: web::Data<NGramTrie>, smoothing: web::Data<ModifiedBackoffKneserNey>) -> impl Responder {
    let mut probabilities = trie.get_prediction_probabilities(smoothing.as_ref(), &req.history);

    probabilities.sort_by_key(|k| k.0);

    let response = PredictionResponse {
        probabilities: probabilities,
    };
    web::Json(response)
}

#[tokio::main]
async fn start_http_server(trie: Arc<NGramTrie>, smoothing: Arc<ModifiedBackoffKneserNey>) -> std::io::Result<()> {
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

fn main() {
    //run_performance_tests("tokens.json");

    NGramTrie::estimate_time_and_ram(475_000_000);
    
    let tokens = NGramTrie::load_json("../cleaned_tokens.json", None).unwrap();

    let mut trie = NGramTrie::fit(tokens, 7, Some(2_usize.pow(14)), None);

    trie.save("../trie_7_475m_v3.bin");

    //let mut trie = NGramTrie::load("../trie_7_475m.bin").unwrap();

    let smoothing = ModifiedBackoffKneserNey::new(&trie);
    
    let probabilities = trie.get_prediction_probabilities(&smoothing, &vec![510, 224, 290, 185, 1528, 135]);
    
    println!("----- Getting rule count -----");
    let rule = NGramTrie::_preprocess_rule_context(&vec![510, 224, 290, 185, 1528, 135], Some("++*+***"));
    let start = Instant::now();
    let count = trie.get_count(&rule);
    let elapsed = start.elapsed();
    println!("Count: {}", count);
    println!("Time taken: {:?}", elapsed);

    let trie = Arc::new(trie);
    let smoothing = Arc::new(smoothing);
}