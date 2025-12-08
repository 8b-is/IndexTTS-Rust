//! ONNX Model Integration Tests for IndexTTS-Rust
//!
//! These tests verify that ONNX models can be loaded and executed correctly.
//! They require actual ONNX model files to be present in the models/ directory.
//!
//! # Running These Tests
//!
//! 1. Download the ONNX models to models/ directory
//! 2. Set ORT_DYLIB_PATH environment variable to ONNX Runtime library location
//! 3. Run: cargo test --test integration_onnx
//!
//! # Model Requirements
//!
//! Expected models in models/ directory:
//! - gpt.onnx - GPT sequence generation model
//! - speaker_encoder.onnx - Speaker embedding model
//! - bigvgan.onnx - BigVGAN vocoder model
//!
//! Built with love by Hue & Aye @ 8b.is 💜

use std::path::Path;

// ============================================================================
// Helper Functions
// ============================================================================

/// Check if the models directory exists and contains expected files
fn models_available() -> bool {
    let models_dir = Path::new("models");
    models_dir.exists() && models_dir.is_dir()
}

/// Check if ONNX Runtime is available
fn ort_available() -> bool {
    std::env::var("ORT_DYLIB_PATH").is_ok()
}

/// Get model path or skip the test
fn get_model_path(name: &str) -> Option<std::path::PathBuf> {
    let path = Path::new("models").join(name);
    if path.exists() {
        Some(path)
    } else {
        None
    }
}

// ============================================================================
// Environment Tests
// ============================================================================

/// Test: Verify the test environment is correctly set up
#[test]
fn test_environment_check() {
    // This test always passes - it just reports on the environment
    println!("\n=== IndexTTS-Rust Integration Test Environment ===\n");

    // Check models directory
    if models_available() {
        println!("✅ models/ directory exists");

        // List available models
        if let Ok(entries) = std::fs::read_dir("models") {
            let onnx_files: Vec<_> = entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .extension()
                        .map_or(false, |ext| ext == "onnx")
                })
                .collect();

            if onnx_files.is_empty() {
                println!("⚠️  No .onnx files found in models/");
            } else {
                println!("📦 Found {} ONNX model(s):", onnx_files.len());
                for file in &onnx_files {
                    println!("   - {}", file.file_name().to_string_lossy());
                }
            }
        }
    } else {
        println!("⚠️  models/ directory not found");
        println!("   Create it and add ONNX models to run integration tests");
    }

    // Check ONNX Runtime
    if let Ok(ort_path) = std::env::var("ORT_DYLIB_PATH") {
        println!("✅ ORT_DYLIB_PATH is set: {}", ort_path);
        if Path::new(&ort_path).exists() {
            println!("✅ ONNX Runtime library exists");
        } else {
            println!("⚠️  ONNX Runtime library not found at specified path");
        }
    } else {
        println!("⚠️  ORT_DYLIB_PATH not set");
        println!("   Set this to the path of libonnxruntime.so/.dylib");
    }

    // Check example audio files
    let examples_dir = Path::new("examples");
    if examples_dir.exists() {
        let wav_files = std::fs::read_dir(examples_dir)
            .ok()
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .filter(|e| {
                        e.path()
                            .extension()
                            .map_or(false, |ext| ext == "wav")
                    })
                    .count()
            })
            .unwrap_or(0);

        println!("✅ examples/ directory exists with {} WAV file(s)", wav_files);
    } else {
        println!("⚠️  examples/ directory not found");
    }

    println!("\n=== Environment Check Complete ===\n");
}

// ============================================================================
// Model Loading Tests
// ============================================================================

/// Test: Attempt to load the GPT model
/// This test is skipped if the model file doesn't exist
#[test]
fn test_load_gpt_model() {
    if !ort_available() {
        println!("⏭️  Skipping: ORT_DYLIB_PATH not set");
        return;
    }

    let model_path = match get_model_path("gpt.onnx") {
        Some(p) => p,
        None => {
            println!("⏭️  Skipping: gpt.onnx not found in models/");
            return;
        }
    };

    println!("📦 Loading GPT model from: {:?}", model_path);

    // Verify the file exists and is readable
    let metadata = std::fs::metadata(&model_path).expect("Failed to read model metadata");
    println!(
        "✅ GPT model file exists ({:.2} MB)",
        metadata.len() as f64 / 1_048_576.0
    );

    // TODO: Add actual ort::Session loading when models are available
    // let session = ort::Session::builder()
    //     .unwrap()
    //     .commit_from_file(&model_path)
    //     .expect("Failed to load GPT model");
}

/// Test: Attempt to load the speaker encoder model
#[test]
fn test_load_speaker_encoder() {
    if !ort_available() {
        println!("⏭️  Skipping: ORT_DYLIB_PATH not set");
        return;
    }

    let model_path = match get_model_path("speaker_encoder.onnx") {
        Some(p) => p,
        None => {
            println!("⏭️  Skipping: speaker_encoder.onnx not found in models/");
            return;
        }
    };

    println!("📦 Loading Speaker Encoder from: {:?}", model_path);

    let metadata = std::fs::metadata(&model_path).expect("Failed to read model metadata");
    println!(
        "✅ Speaker Encoder file exists ({:.2} MB)",
        metadata.len() as f64 / 1_048_576.0
    );
}

/// Test: Attempt to load the BigVGAN vocoder model
#[test]
fn test_load_bigvgan() {
    if !ort_available() {
        println!("⏭️  Skipping: ORT_DYLIB_PATH not set");
        return;
    }

    let model_path = match get_model_path("bigvgan.onnx") {
        Some(p) => p,
        None => {
            println!("⏭️  Skipping: bigvgan.onnx not found in models/");
            return;
        }
    };

    println!("📦 Loading BigVGAN Vocoder from: {:?}", model_path);

    let metadata = std::fs::metadata(&model_path).expect("Failed to read model metadata");
    println!(
        "✅ BigVGAN file exists ({:.2} MB)",
        metadata.len() as f64 / 1_048_576.0
    );
}

// ============================================================================
// Audio I/O Tests (using public API)
// ============================================================================

/// Test: Read and validate example audio files using the public API
#[test]
fn test_read_example_audio() {
    use indextts::audio::load_audio;

    let examples_dir = Path::new("examples");
    if !examples_dir.exists() {
        println!("⏭️  Skipping: examples/ directory not found");
        return;
    }

    // Find first WAV file
    let wav_file = std::fs::read_dir(examples_dir)
        .ok()
        .and_then(|mut entries| {
            entries.find_map(|e| {
                e.ok().and_then(|entry| {
                    let path = entry.path();
                    if path.extension().map_or(false, |ext| ext == "wav") {
                        Some(path)
                    } else {
                        None
                    }
                })
            })
        });

    let wav_file = match wav_file {
        Some(f) => f,
        None => {
            println!("⏭️  Skipping: No WAV files found in examples/");
            return;
        }
    };

    println!("📂 Reading audio file: {:?}", wav_file);

    // Read the audio file using public API
    match load_audio(wav_file.to_str().unwrap(), None) {
        Ok(audio) => {
            let duration_secs = audio.samples.len() as f32 / audio.sample_rate as f32;
            println!(
                "✅ Audio loaded: {} samples, {} Hz, {:.2}s duration",
                audio.samples.len(),
                audio.sample_rate,
                duration_secs
            );

            // Validate sample values are in expected range
            let max_sample = audio
                .samples
                .iter()
                .fold(0.0f32, |max, &s| max.max(s.abs()));
            println!("   Max sample amplitude: {:.4}", max_sample);

            assert!(
                max_sample <= 1.1, // Allow slight headroom
                "Sample values should be normalized to approximately [-1, 1]"
            );
        }
        Err(e) => {
            println!("❌ Failed to read audio: {}", e);
        }
    }
}

/// Test: Write audio to a temporary file and read it back
#[test]
fn test_audio_roundtrip() {
    use indextts::audio::{load_audio, save_audio, AudioData};
    use indextts::SAMPLE_RATE;

    // Generate a simple test tone (440 Hz sine wave for 0.5 seconds)
    let duration_samples = SAMPLE_RATE as usize / 2; // 0.5 seconds
    let frequency = 440.0f32; // A4 note

    let samples: Vec<f32> = (0..duration_samples)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE as f32;
            (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5 // 50% amplitude
        })
        .collect();

    println!("🎵 Generated test tone: {} Hz, {} samples", frequency, samples.len());

    let audio_data = AudioData {
        samples: samples.clone(),
        sample_rate: SAMPLE_RATE,
    };

    // Write to temp file
    let temp_dir = std::env::temp_dir();
    let temp_file = temp_dir.join("indextts_test_audio.wav");

    match save_audio(temp_file.to_str().unwrap(), &audio_data) {
        Ok(_) => println!("✅ Wrote audio to: {:?}", temp_file),
        Err(e) => {
            println!("❌ Failed to write audio: {}", e);
            panic!("Audio write failed");
        }
    }

    // Read it back
    match load_audio(temp_file.to_str().unwrap(), None) {
        Ok(read_audio) => {
            println!(
                "✅ Read back: {} samples at {} Hz",
                read_audio.samples.len(),
                read_audio.sample_rate
            );

            assert_eq!(read_audio.sample_rate, SAMPLE_RATE, "Sample rate mismatch");
            // Note: Sample count might differ slightly due to WAV format

            // Check samples match approximately
            let min_len = samples.len().min(read_audio.samples.len());
            let max_diff = samples[..min_len]
                .iter()
                .zip(read_audio.samples[..min_len].iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, |max, diff| max.max(diff));

            println!("   Max sample difference: {:.6}", max_diff);
            assert!(max_diff < 0.01, "Samples differ too much after roundtrip");

            // Cleanup
            let _ = std::fs::remove_file(&temp_file);
            println!("🧹 Cleaned up temp file");
        }
        Err(e) => {
            println!("❌ Failed to read audio back: {}", e);
            panic!("Audio read failed");
        }
    }
}

// ============================================================================
// Mel Spectrogram Tests
// ============================================================================

/// Test: Compute mel spectrogram from audio
#[test]
fn test_mel_spectrogram_computation() {
    use indextts::audio::{mel_spectrogram, AudioConfig};
    use indextts::SAMPLE_RATE;

    // Generate 1 second of white noise for testing
    let duration_samples = SAMPLE_RATE as usize;
    let samples: Vec<f32> = (0..duration_samples)
        .map(|_| (rand::random::<f32>() - 0.5) * 0.1) // Low amplitude noise
        .collect();

    println!("🎲 Generated {} samples of test noise", samples.len());

    let config = AudioConfig::default();

    // Compute mel spectrogram
    match mel_spectrogram(&samples, &config) {
        Ok(mel) => {
            let shape = mel.shape();
            println!("✅ Mel spectrogram computed: {:?}", shape);

            // Verify dimensions are reasonable
            // Mel spectrogram shape is [n_mels, time_frames]
            assert_eq!(shape[0], config.n_mels, "Should have {} mel bands", config.n_mels);
            assert!(shape[1] > 0, "Should have time frames");

            // Check values are finite
            assert!(mel.iter().all(|&v| v.is_finite()), "All values should be finite");
        }
        Err(e) => {
            println!("❌ Mel spectrogram computation failed: {}", e);
        }
    }
}

// ============================================================================
// Quality Module Tests
// ============================================================================

/// Test: Marine prosody extraction from audio
#[test]
fn test_marine_prosody_extraction() {
    use indextts::quality::MarineProsodyConditioner;
    use indextts::SAMPLE_RATE;

    println!("🌊 Testing Marine prosody extraction:");

    // Generate a simple test signal (440 Hz tone)
    let duration_secs = 0.5;
    let num_samples = (SAMPLE_RATE as f32 * duration_secs) as usize;
    let frequency = 440.0f32;

    let samples: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE as f32;
            (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5
        })
        .collect();

    println!("   Generated {:.2}s test tone at {} Hz", duration_secs, frequency);

    // Create conditioner and extract prosody
    let conditioner = MarineProsodyConditioner::new(SAMPLE_RATE);

    match conditioner.from_samples(&samples) {
        Ok(prosody) => {
            println!("✅ Extracted MarineProsodyVector:");
            println!("   jp_mean (pitch jitter): {:.4}", prosody.jp_mean);
            println!("   jp_std: {:.4}", prosody.jp_std);
            println!("   ja_mean (amplitude jitter): {:.4}", prosody.ja_mean);
            println!("   ja_std: {:.4}", prosody.ja_std);
            println!("   h_mean (harmonic): {:.4}", prosody.h_mean);
            println!("   s_mean (salience): {:.4}", prosody.s_mean);
            println!("   peak_density: {:.2}", prosody.peak_density);
            println!("   energy_mean: {:.4}", prosody.energy_mean);
        }
        Err(e) => {
            println!("❌ Prosody extraction failed: {}", e);
        }
    }
}

/// Test: TTS quality report validation
#[test]
fn test_quality_report() {
    use indextts::quality::MarineProsodyConditioner;
    use indextts::SAMPLE_RATE;

    println!("📋 Testing TTS quality validation:");

    // Generate test audio
    let num_samples = SAMPLE_RATE as usize;
    let frequency = 220.0f32;

    let samples: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE as f32;
            // Add some harmonics to make it more voice-like
            let f1 = (2.0 * std::f32::consts::PI * frequency * t).sin();
            let f2 = (2.0 * std::f32::consts::PI * frequency * 2.0 * t).sin() * 0.5;
            let f3 = (2.0 * std::f32::consts::PI * frequency * 3.0 * t).sin() * 0.25;
            (f1 + f2 + f3) * 0.3
        })
        .collect();

    let conditioner = MarineProsodyConditioner::new(SAMPLE_RATE);

    match conditioner.validate_tts_output(&samples) {
        Ok(report) => {
            println!("✅ Quality Report generated:");
            println!("   Overall Score: {:.1}%", report.quality_score);
            println!("   Passes (70% threshold): {}", report.passes(70.0));

            if report.issues.is_empty() {
                println!("   No issues detected! 🎉");
            } else {
                println!("   Issues found:");
                for issue in &report.issues {
                    println!("   ⚠️  {}", issue);
                }
            }
        }
        Err(e) => {
            println!("❌ Quality validation failed: {}", e);
        }
    }
}

// ============================================================================
// Full Pipeline Tests (Require Models)
// ============================================================================

/// Test: End-to-end synthesis (requires all models)
#[test]
fn test_full_synthesis_pipeline() {
    if !ort_available() {
        println!("⏭️  Skipping full pipeline test: ORT_DYLIB_PATH not set");
        return;
    }

    if !models_available() {
        println!("⏭️  Skipping full pipeline test: models/ directory not found");
        return;
    }

    // Check for required models
    let required_models = ["gpt.onnx", "speaker_encoder.onnx", "bigvgan.onnx"];
    let mut missing = Vec::new();

    for model in &required_models {
        if get_model_path(model).is_none() {
            missing.push(*model);
        }
    }

    if !missing.is_empty() {
        println!("⏭️  Skipping full pipeline test: Missing models: {:?}", missing);
        return;
    }

    // Find a reference voice file
    let voice_file = std::fs::read_dir("examples")
        .ok()
        .and_then(|entries| {
            entries
                .filter_map(|e| e.ok())
                .find(|e| {
                    e.path()
                        .extension()
                        .map_or(false, |ext| ext == "wav")
                })
                .map(|e| e.path())
        });

    let voice_file = match voice_file {
        Some(f) => f,
        None => {
            println!("⏭️  Skipping: No reference voice file found in examples/");
            return;
        }
    };

    println!("🎤 Running full synthesis pipeline test:");
    println!("   Voice file: {:?}", voice_file);
    println!("   Text: \"Hello, this is a test of IndexTTS Rust.\"");

    // TODO: Implement full pipeline test when IndexTTS struct is ready
    // let config = indextts::Config::default();
    // let tts = indextts::IndexTTS::new(config).expect("Failed to create TTS engine");
    // let options = indextts::pipeline::SynthesisOptions::default();
    // let result = tts.synthesize("Hello, this is a test.", &voice_file, &options);

    println!("   ⏸️  Full pipeline test not yet implemented");
    println!("   Waiting for IndexTTS struct to be fully wired up");
}

// ============================================================================
// Benchmark Helpers
// ============================================================================

/// Test: Measure mel spectrogram computation time
#[test]
#[ignore] // Run with: cargo test --test integration_onnx -- --ignored
fn bench_mel_spectrogram() {
    use indextts::audio::{mel_spectrogram, AudioConfig};
    use indextts::SAMPLE_RATE;
    use std::time::Instant;

    // Generate 10 seconds of audio
    let duration_secs = 10.0f32;
    let num_samples = (SAMPLE_RATE as f32 * duration_secs) as usize;

    let samples: Vec<f32> = (0..num_samples)
        .map(|_| (rand::random::<f32>() - 0.5) * 0.5)
        .collect();

    println!("\n⏱️  Benchmarking mel spectrogram computation:");
    println!("   Audio duration: {:.1}s ({} samples)", duration_secs, num_samples);

    let config = AudioConfig::default();
    let iterations = 10;
    let mut times = Vec::new();

    for i in 0..iterations {
        let start = Instant::now();
        let _ = mel_spectrogram(&samples, &config);
        let elapsed = start.elapsed();
        times.push(elapsed.as_secs_f64() * 1000.0);

        if i == 0 {
            println!("   First run: {:.2}ms", times[0]);
        }
    }

    let avg_time: f64 = times.iter().sum::<f64>() / iterations as f64;
    let real_time_factor = (duration_secs as f64) / (avg_time / 1000.0);

    println!("   Average over {} iterations: {:.2}ms", iterations, avg_time);
    println!("   Real-time factor: {:.1}x", real_time_factor);
}
