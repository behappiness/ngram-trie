#![allow(warnings)]
pub mod trie;
pub mod smoothing;
pub mod smoothed_trie;

use trie::{NGramTrie, trienode::TrieNode, CACHE_C, CACHE_N};
use smoothing::{ModifiedBackoffKneserNey, CACHE_S};
use sorted_vector_map::SortedVectorMap;
use smoothed_trie::SmoothedTrie;

use rclite::Arc;
use serde::Serialize;
use serde::Deserialize;
use std::time::Instant;
use std::fs::OpenOptions;
use std::io::Write;
use actix_web::{web, App, HttpServer, Responder};
use log::{info, debug, error};

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
            let trie = NGramTrie::fit(tokens.clone(), *n_gram_length, 2_usize.pow(14), Some(data_size));
            let fit_time = start.elapsed().as_secs_f64(); 
            // Measure RAM usage
            let ram_usage = 0 as f64 / (1024.0 * 1024.0);

            // Write statistics to file
            writeln!(
                file,
                "{},{},{},{:.2}",
                data_size, n_gram_length, fit_time, ram_usage
            ).unwrap();

            println!(
                "Completed: Data Size = {}, N-gram Length = {}, Fit Time = {:.2}, RAM Usage = {:.2} MB",
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
struct UnsmoothedProbabilityRequest {
    history: Vec<u16>,
}

#[derive(Serialize)]
struct UnsmoothedProbabilityResponse {
    probabilities: Vec<(String, Vec<f64>)>,
}

async fn get_unsmoothed_probabilities(
    req: web::Json<UnsmoothedProbabilityRequest>, 
    smoothed_trie: web::Data<Arc<SmoothedTrie>>
) -> impl Responder {
    let probabilities = smoothed_trie.get_unsmoothed_probabilities(&req.history);
    web::Json(UnsmoothedProbabilityResponse { probabilities })
}

#[derive(Serialize, Deserialize)]
struct SmoothedProbabilityRequest {
    history: Vec<u16>,
    rule_set: Option<Vec<String>>,
}

#[derive(Serialize)]
struct SmoothedProbabilityResponse {
    probabilities: Vec<(String, Vec<f64>)>,
}

async fn get_smoothed_probabilities(
    req: web::Json<SmoothedProbabilityRequest>, 
    smoothed_trie: web::Data<Arc<SmoothedTrie>>
) -> impl Responder {
    let probabilities = smoothed_trie.get_smoothed_probabilities(&req.history, req.rule_set.clone());
    web::Json(SmoothedProbabilityResponse { probabilities })
}

#[tokio::main]
async fn start_http_server(smoothed_trie: SmoothedTrie) -> std::io::Result<()> {
    let server_workers = 2;
    println!("----- Starting HTTP server with {} workers -----", server_workers);
    
    // Create the Data wrapper once, outside the HttpServer::new closure
    let shared_trie = web::Data::new(Arc::new(smoothed_trie));
    
    HttpServer::new(move || {
        App::new()
            .app_data(shared_trie.clone()) // Clone the Data wrapper, not the trie itself
            .service(web::resource("/unsmoothed_predict").route(web::post().to(get_unsmoothed_probabilities)))
            .service(web::resource("/smoothed_predict").route(web::post().to(get_smoothed_probabilities)))
    })
    .workers(server_workers)
    .bind("127.0.0.1:8080")?
    .run()
    .await
}

fn main() {
    env_logger::Builder::new()
        .format(|buf, record| {
            writeln!(
                buf,
                "{} [{}] {}",
                chrono::Local::now().format("%H:%M:%S"),
                record.level(),
                record.args()
            )
        })
        .filter_level(log::LevelFilter::Debug)
        .init();
    //run_performance_tests("tokens.json");
    //NGramTrie::estimate_time_and_ram(475_000_000);
    
    let mut smoothed_trie = SmoothedTrie::new(NGramTrie::new(9, 2_usize.pow(14)), None);

    // let tokens = NGramTrie::load_json("../170k_tokens.json", None).unwrap();
    // smoothed_trie.fit(tokens, 8, 2_usize.pow(14), None, Some("modified_kneser_ney".to_string()));

    // smoothed_trie.save("trie");

    smoothed_trie.load("ngram");

    // info!("----- Getting rule count -----");
    // let rule = NGramTrie::_preprocess_rule_context(&vec![987, 4015, 935, 2940, 3947, 987, 4015], Some("+++*++*"));
    // let start = Instant::now();
    // let count = smoothed_trie.get_count(rule.clone());
    // let elapsed = start.elapsed();
    // info!("Count: {}", count);
    // info!("Time taken: {:.2?}", elapsed);
    
    // 170k_tokens
    // let history = vec![987, 4015, 935, 2940, 3947, 987, 4015, 3042, 652, 987, 3211, 278, 4230];
    // let history = vec![987, 4015, 935, 2940, 3947, 987, 4015];
    // smoothed_trie.set_all_ruleset_by_length(7);
    // let probabilities = smoothed_trie.get_smoothed_probabilities(&history, None);

    // for (rule, token_probs) in &probabilities {
    //     let total_prob: f64 = token_probs.iter().map(|prob| prob).sum();
    //     println!("Rule: {}, Total Probability: {:.6}", rule, total_prob);
    // }
    // for p in probabilities[369].1.iter() {
    //     print!("{:.5} ", p);
    // }
    // println!("argmax: {:?}", probabilities[369].1.iter().max_by(|a, b| a.partial_cmp(b).unwrap()));
    
    // smoothed_trie.set_all_ruleset_by_length(7);
    // smoothed_trie.fit_smoothing(Some("modified_kneser_ney".to_string()));

    // // 475m_tokens
    // //let history = vec![157, 973, 712, 132, 3618, 237, 132, 4988, 134, 234, 342, 330, 4389, 3143];
    // //test_seq_smoothing(&mut smoothed_trie, history);
    // // smoothed_trie.get_prediction_probabilities(&vec![987, 4015, 935, 2940, 3947, 987, 4015]);
    // let data = vec![
    //     173, 0, 2, 8661, 3609, 15, 2270, 1432, 705, 349, 277, 213, 17, 814, 4347, 8661, 3609, 237, 10228, 2266, 238, 215, 719, 1432, 1096, 1284, 286, 444, 2625, 238, 719, 1432, 13002, 377, 11064, 2075, 720, 240, 5093, 17, 264, 1873, 1427, 413, 287, 215, 1263, 15, 4547, 237, 3685, 238, 215, 4164, 950, 1133, 270, 215, 2680, 1073, 1789, 238, 3594, 3544, 2401, 1000, 1139, 237, 1934, 17, 5518, 229, 240, 215, 5755, 226, 839, 3609, 238, 215, 4164, 15, 215, 1991, 9570, 413, 287, 4531, 270, 1284, 270, 397, 1270, 238, 215, 950, 312, 769, 312, 1652, 229, 2183, 294, 209, 2509, 866, 225, 17, 1224, 2183, 312, 2106, 279, 411, 4104, 89, 755, 279, 411, 1589, 350, 235, 5898, 15, 3279, 237, 9644, 229, 3643, 4144, 238, 1096, 1690, 279, 3, 173, 0, 173, 0, 2, 4108, 2098, 43, 15, 15064, 15, 2714, 370, 1247, 15, 3349, 229, 10055, 415, 279, 1030, 240, 8398, 2095, 328, 3689, 17, 14001, 7218, 572, 7843, 12756, 4745, 17, 13291, 306, 1671, 620, 3546, 1077, 4831, 240, 1884, 12756, 4745, 863, 515, 17, 561, 5645, 238, 209, 12741, 452, 8295, 1685, 16052, 12756, 4745, 294, 382, 5022, 15, 10407, 374, 209, 2434, 2119, 2544, 209, 9208, 4297, 17, 825, 670, 7187, 515, 209, 1298, 15551, 2119, 237, 1242, 328, 286,
    // ];

    // let start = Instant::now();
    // for i in 0..(data.len() - 7) {
    //     let context = &data[i..i + 7];
    //     smoothed_trie.get_unsmoothed_probabilities(context);
    // }
    // let elapsed = start.elapsed();
    // info!("Time taken for 32 random context predictions: {:.2?}", elapsed);

    let root_count = smoothed_trie.trie.root.count;
    println!("Root node count (tokens - ngram length + 1): {}", root_count);

    let branching_factors = smoothed_trie.average_branching_factor_per_layer();
    println!("Average branching factors per layer:");
    for (i, factor) in branching_factors.iter().enumerate() {
        println!("Layer {}: {:.2}", i, factor);
    }

    println!("\nNodes per layer:");
    let mut total_nodes = 1;
    println!("Layer {}: {} nodes", 0, total_nodes);
    for layer in 1..=smoothed_trie.trie.n_gram_max_length {
        // using root node so it doesn't cache anything
        let nodes = smoothed_trie.trie.root.find_all_nodes(&vec![None; layer as usize]).len();
        total_nodes += nodes;
        println!("Layer {}: {} nodes", layer, nodes);
    }
    println!("\nTotal number of nodes in tree: {}", total_nodes);

    let mut cumprod = Vec::with_capacity(branching_factors.len());
    let mut running_product = 1.0;
    for &factor in branching_factors.iter() {
        running_product *= factor;
        cumprod.push(running_product);
    }
    println!("Sum of cumulative product of branching factors: {:.2?}", cumprod.into_iter().sum::<f64>() + 1.0);


    // for _ in 0..100 {
    //     test_seq_smoothing(&mut smoothed_trie, history.clone());
    //     smoothed_trie.reset_cache();
    // }


    start_http_server(smoothed_trie).unwrap();
}

fn test_seq_smoothing(smoothed_trie: &mut SmoothedTrie, history: Vec<u16>) {
    info!("----- Testing smoothing -----");
    let start = Instant::now();
    for i in 0..history.len() - smoothed_trie.trie.n_gram_max_length as usize + 1 {
        let _history = history[i..i + smoothed_trie.trie.n_gram_max_length as usize - 1].to_vec();
        let unsmoothed_probabilities = smoothed_trie.get_unsmoothed_probabilities(&_history);
        let smoothed_probabilities = smoothed_trie.get_smoothed_probabilities(&_history, None);
        //smoothed_trie.debug_cache_sizes();
    }
    let elapsed = start.elapsed();
    info!("Time taken for {} context predictions: {:.2?}", history.len() - smoothed_trie.trie.n_gram_max_length as usize + 1, elapsed);
}
