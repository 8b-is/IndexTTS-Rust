//! Benchmark for model inference

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use indextts::model::{sample_from_logits, SamplingStrategy};
use indextts::text::{TextNormalizer, TextTokenizer, TokenizerConfig};

fn bench_sampling(c: &mut Criterion) {
    let vocab_size = 8194;
    let logits: Vec<f32> = (0..vocab_size).map(|i| (i as f32 / 1000.0).sin()).collect();

    c.bench_function("greedy_sampling", |b| {
        b.iter(|| {
            sample_from_logits(black_box(&logits), black_box(&SamplingStrategy::Greedy))
        })
    });

    c.bench_function("top_k_sampling", |b| {
        b.iter(|| {
            sample_from_logits(
                black_box(&logits),
                black_box(&SamplingStrategy::TopK { k: 50 }),
            )
        })
    });

    c.bench_function("top_p_sampling", |b| {
        b.iter(|| {
            sample_from_logits(
                black_box(&logits),
                black_box(&SamplingStrategy::TopP { p: 0.95 }),
            )
        })
    });

    c.bench_function("top_kp_sampling", |b| {
        b.iter(|| {
            sample_from_logits(
                black_box(&logits),
                black_box(&SamplingStrategy::TopKP { k: 50, p: 0.95 }),
            )
        })
    });
}

fn bench_text_processing(c: &mut Criterion) {
    let normalizer = TextNormalizer::new();
    let tokenizer = TextTokenizer::new(TokenizerConfig::default()).unwrap();

    let english_text = "Hello world, this is a test of the text-to-speech system.";
    let chinese_text = "你好世界，这是一个语音合成测试。";
    let mixed_text = "Hello 世界, this is 测试 of TTS.";

    c.bench_function("normalize_english", |b| {
        b.iter(|| normalizer.normalize(black_box(english_text)))
    });

    c.bench_function("normalize_chinese", |b| {
        b.iter(|| normalizer.normalize(black_box(chinese_text)))
    });

    c.bench_function("normalize_mixed", |b| {
        b.iter(|| normalizer.normalize(black_box(mixed_text)))
    });

    c.bench_function("tokenize_english", |b| {
        b.iter(|| tokenizer.encode(black_box(english_text)))
    });

    c.bench_function("tokenize_chinese", |b| {
        b.iter(|| tokenizer.encode(black_box(chinese_text)))
    });

    c.bench_function("tokenize_mixed", |b| {
        b.iter(|| tokenizer.encode(black_box(mixed_text)))
    });
}

fn bench_vocoder(c: &mut Criterion) {
    use indextts::vocoder::{create_bigvgan_22k, Vocoder};
    use ndarray::Array2;

    let vocoder = create_bigvgan_22k();

    // Small mel (10 frames ~ 0.25s)
    let small_mel = Array2::zeros((80, 10));
    c.bench_function("vocoder_small", |b| {
        b.iter(|| vocoder.synthesize(black_box(&small_mel)))
    });

    // Medium mel (100 frames ~ 2.5s)
    let medium_mel = Array2::zeros((80, 100));
    c.bench_function("vocoder_medium", |b| {
        b.iter(|| vocoder.synthesize(black_box(&medium_mel)))
    });
}

criterion_group!(benches, bench_sampling, bench_text_processing, bench_vocoder);
criterion_main!(benches);
