//! Benchmark for mel-spectrogram computation

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use indextts::audio::{mel_spectrogram, AudioConfig};

fn bench_mel_spectrogram(c: &mut Criterion) {
    let config = AudioConfig::default();

    // Generate 1 second of audio
    let num_samples = config.sample_rate as usize;
    let signal: Vec<f32> = (0..num_samples).map(|i| (i as f32 * 0.01).sin()).collect();

    c.bench_function("mel_spectrogram_1s", |b| {
        b.iter(|| mel_spectrogram(black_box(&signal), black_box(&config)))
    });

    // Generate 10 seconds of audio
    let long_signal: Vec<f32> = (0..num_samples * 10)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();

    c.bench_function("mel_spectrogram_10s", |b| {
        b.iter(|| mel_spectrogram(black_box(&long_signal), black_box(&config)))
    });
}

fn bench_stft(c: &mut Criterion) {
    let config = AudioConfig::default();
    let num_samples = config.sample_rate as usize;
    let signal: Vec<f32> = (0..num_samples).map(|i| (i as f32 * 0.01).sin()).collect();

    c.bench_function("stft_1s", |b| {
        b.iter(|| {
            indextts::audio::mel::stft(
                black_box(&signal),
                black_box(config.n_fft),
                black_box(config.hop_length),
                black_box(config.win_length),
            )
        })
    });
}

criterion_group!(benches, bench_mel_spectrogram, bench_stft);
criterion_main!(benches);
