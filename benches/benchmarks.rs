use criterion::{criterion_group, criterion_main, Criterion};
use ngram_trie::trie::NGramTrie;
use ngram_trie::smoothed_trie::SmoothedTrie;

fn bench_modified_kneser_ney_smoothing(c: &mut Criterion) {
    let tokens = NGramTrie::load_json("../170k_tokens.json", None).unwrap();
    let trie = NGramTrie::new(8, 2_usize.pow(14));
    let mut smoothed_trie = SmoothedTrie::new(trie, None);
    smoothed_trie.fit(tokens, 8, 2_usize.pow(14), None, Some("modified_kneser_ney".to_string()));

    let history = vec![987, 4015, 935, 2940, 3947, 987, 4015];

    c.bench_function("modified_kneser_ney_smoothing", |b| {
        b.iter_batched(
            || {
                // Reset the cache before each run
                smoothed_trie.reset_cache();
                history.clone()
            },
            |history| {
                smoothed_trie.get_smoothed_probabilities(&history);
            },
            criterion::BatchSize::NumIterations(1), // Set sample size to 3
        );
    });
}

fn bench_stupid_backoff_smoothing(c: &mut Criterion) {
    let tokens = NGramTrie::load_json("../170k_tokens.json", None).unwrap();
    let trie = NGramTrie::new(8, 2_usize.pow(14));
    let mut smoothed_trie = SmoothedTrie::new(trie, None);
    smoothed_trie.fit(tokens, 8, 0, None, Some("stupid_backoff".to_string()));

    let history = vec![987, 4015, 935, 2940, 3947, 987, 4015];

    c.bench_function("stupid_backoff_smoothing", |b| {
        b.iter_batched(
            || {
                // Reset the cache before each run
                smoothed_trie.reset_cache();
                history.clone()
            },
            |history| {
                smoothed_trie.get_smoothed_probabilities(&history);
            },
            criterion::BatchSize::NumIterations(1), // Set sample size to 3
        );
    });
}

fn configure_criterion() -> Criterion {
    Criterion::default()
        .measurement_time(std::time::Duration::new(500, 0)) // Set measurement time to 5 seconds
        .sample_size(10) // Set sample size to 10
}

criterion_group!{
    name = benches;
    config = configure_criterion();
    targets = bench_modified_kneser_ney_smoothing, bench_stupid_backoff_smoothing
}
criterion_main!(benches);