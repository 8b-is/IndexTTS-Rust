//! IndexTTS CLI - High-performance Text-to-Speech in Rust
//!
//! Command-line interface for IndexTTS synthesizer

use clap::{Parser, Subcommand};
use indextts::{
    pipeline::{IndexTTS, SynthesisOptions},
    Config, Result,
};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "indextts",
    about = "High-performance Text-to-Speech engine in Rust",
    version,
    author
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Synthesize speech from text
    Synthesize {
        /// Text to synthesize
        #[arg(short, long)]
        text: String,

        /// Speaker reference audio file
        #[arg(short = 'v', long)]
        voice: PathBuf,

        /// Output audio file path
        #[arg(short, long, default_value = "output.wav")]
        output: PathBuf,

        /// Configuration file path
        #[arg(short, long)]
        config: Option<PathBuf>,

        /// Model directory
        #[arg(short, long, default_value = "models")]
        model_dir: PathBuf,

        /// Emotion vector (comma-separated, 8 values 0-1)
        #[arg(long)]
        emotion: Option<String>,

        /// Emotion strength (0-1)
        #[arg(long, default_value = "1.0")]
        emotion_alpha: f32,

        /// Top-k sampling parameter
        #[arg(long, default_value = "50")]
        top_k: usize,

        /// Top-p sampling parameter
        #[arg(long, default_value = "0.95")]
        top_p: f32,

        /// Repetition penalty
        #[arg(long, default_value = "1.1")]
        repetition_penalty: f32,

        /// Use FP16 inference
        #[arg(long)]
        fp16: bool,

        /// Device (cpu, cuda:0, etc.)
        #[arg(short, long, default_value = "cpu")]
        device: String,
    },

    /// Synthesize from a text file
    SynthesizeFile {
        /// Input text file
        #[arg(short, long)]
        input: PathBuf,

        /// Speaker reference audio file
        #[arg(short = 'v', long)]
        voice: PathBuf,

        /// Output audio file path
        #[arg(short, long, default_value = "output.wav")]
        output: PathBuf,

        /// Configuration file path
        #[arg(short, long)]
        config: Option<PathBuf>,

        /// Model directory
        #[arg(short, long, default_value = "models")]
        model_dir: PathBuf,

        /// Silence between segments (milliseconds)
        #[arg(long, default_value = "200")]
        silence_ms: u32,
    },

    /// Generate default configuration file
    InitConfig {
        /// Output path for config file
        #[arg(short, long, default_value = "config.yaml")]
        output: PathBuf,
    },

    /// Show information about the system
    Info,

    /// Run benchmarks
    Benchmark {
        /// Number of iterations
        #[arg(short, long, default_value = "10")]
        iterations: usize,
    },
}

fn main() -> Result<()> {
    // Initialize logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Synthesize {
            text,
            voice,
            output,
            config,
            model_dir,
            emotion,
            emotion_alpha,
            top_k,
            top_p,
            repetition_penalty,
            fp16: _,
            device: _,
        } => {
            log::info!("IndexTTS Synthesizer");
            log::info!("====================");

            // Load or create config
            let cfg = if let Some(config_path) = config {
                Config::load(config_path)?
            } else {
                let mut cfg = Config::default();
                cfg.model_dir = model_dir;
                cfg
            };

            // Create TTS instance
            let tts = IndexTTS::new(cfg)?;

            // Parse emotion vector
            let emotion_vec = emotion.map(|s| {
                s.split(',')
                    .filter_map(|v| v.trim().parse::<f32>().ok())
                    .collect::<Vec<f32>>()
            });

            // Create synthesis options
            let options = SynthesisOptions {
                emotion_vector: emotion_vec,
                emotion_alpha,
                sampling: indextts::model::SamplingStrategy::TopKP { k: top_k, p: top_p },
                repetition_penalty,
                ..Default::default()
            };

            // Synthesize
            log::info!("Text: {}", &text[..text.len().min(100)]);
            log::info!("Voice: {}", voice.display());
            log::info!("Output: {}", output.display());

            let result = tts.synthesize_to_file(
                &text,
                voice.to_str().unwrap(),
                output.to_str().unwrap(),
                &options,
            )?;

            log::info!("Duration: {}", result.duration_formatted());
            log::info!("Processing time: {:.2}s", result.processing_time);
            log::info!("Real-time factor: {:.3}x", result.rtf);

            println!("✓ Synthesis complete: {}", output.display());
        }

        Commands::SynthesizeFile {
            input,
            voice,
            output,
            config,
            model_dir,
            silence_ms,
        } => {
            log::info!("IndexTTS File Synthesizer");
            log::info!("==========================");

            // Read text file
            let text = std::fs::read_to_string(&input)?;

            // Load or create config
            let cfg = if let Some(config_path) = config {
                Config::load(config_path)?
            } else {
                let mut cfg = Config::default();
                cfg.model_dir = model_dir;
                cfg
            };

            // Create TTS instance
            let tts = IndexTTS::new(cfg)?;

            // Create synthesis options
            let options = SynthesisOptions {
                segment_silence_ms: silence_ms,
                ..Default::default()
            };

            // Synthesize
            log::info!("Input file: {}", input.display());
            log::info!("Text length: {} characters", text.len());

            let result = tts.synthesize_long(
                &text,
                voice.to_str().unwrap(),
                &options,
            )?;

            result.save(&output)?;

            log::info!("Duration: {}", result.duration_formatted());
            log::info!("Processing time: {:.2}s", result.processing_time);
            log::info!("Real-time factor: {:.3}x", result.rtf);

            println!("✓ Synthesis complete: {}", output.display());
        }

        Commands::InitConfig { output } => {
            log::info!("Creating default configuration...");

            let config = Config::default();
            config.save(&output)?;

            println!("✓ Configuration saved to: {}", output.display());
        }

        Commands::Info => {
            println!("IndexTTS - High-performance Text-to-Speech Engine");
            println!("==================================================");
            println!("Version: {}", indextts::VERSION);
            println!("Platform: {}", std::env::consts::OS);
            println!("Architecture: {}", std::env::consts::ARCH);
            println!();
            println!("Features:");
            println!("  - Multi-language support (Chinese, English, mixed)");
            println!("  - Zero-shot voice cloning");
            println!("  - 8-dimensional emotion control");
            println!("  - High-quality neural vocoding (BigVGAN)");
            println!("  - SIMD-optimized audio processing");
            println!("  - Parallel processing with Rayon");
            println!();
            println!("Sample Rate: {} Hz", indextts::SAMPLE_RATE);
            println!("Mel Bands: {}", indextts::N_MELS);
            println!("FFT Size: {}", indextts::N_FFT);
            println!("Hop Length: {}", indextts::HOP_LENGTH);
            println!();
            println!("CPU Cores: {}", num_cpus::get());
            println!("Physical Cores: {}", num_cpus::get_physical());
        }

        Commands::Benchmark { iterations } => {
            log::info!("Running benchmarks ({} iterations)...", iterations);

            // Benchmark mel-spectrogram computation
            benchmark_mel_spectrogram(iterations);

            // Benchmark tokenization
            benchmark_tokenization(iterations);

            // Benchmark vocoder
            benchmark_vocoder(iterations);

            println!("✓ Benchmarks complete");
        }
    }

    Ok(())
}

fn benchmark_mel_spectrogram(iterations: usize) {
    use indextts::audio::{mel_spectrogram, AudioConfig};
    use std::time::Instant;

    println!("\nMel-Spectrogram Benchmark");
    println!("-------------------------");

    let config = AudioConfig::default();
    let num_samples = config.sample_rate as usize; // 1 second of audio
    let signal: Vec<f32> = (0..num_samples)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = mel_spectrogram(&signal, &config);
    }
    let elapsed = start.elapsed();

    let per_iter = elapsed.as_secs_f32() / iterations as f32;
    println!("  Signal length: {} samples ({:.2}s)", num_samples, num_samples as f32 / config.sample_rate as f32);
    println!("  Iterations: {}", iterations);
    println!("  Total time: {:.3}s", elapsed.as_secs_f32());
    println!("  Per iteration: {:.3}ms", per_iter * 1000.0);
    println!("  Throughput: {:.1}x real-time", 1.0 / per_iter);
}

fn benchmark_tokenization(iterations: usize) {
    use indextts::text::{TextNormalizer, TextTokenizer, TokenizerConfig};
    use std::time::Instant;

    println!("\nTokenization Benchmark");
    println!("----------------------");

    let normalizer = TextNormalizer::new();
    let tokenizer = TextTokenizer::new(TokenizerConfig::default()).unwrap();

    let test_texts = vec![
        "Hello world, this is a test of the text-to-speech system.",
        "The quick brown fox jumps over the lazy dog.",
        "你好世界，这是一个测试。",
        "Mixed language: Hello 世界 and 你好 world.",
    ];

    let start = Instant::now();
    for _ in 0..iterations {
        for text in &test_texts {
            let normalized = normalizer.normalize(text).unwrap();
            let _tokens = tokenizer.encode(&normalized).unwrap();
        }
    }
    let elapsed = start.elapsed();

    let total_chars: usize = test_texts.iter().map(|t| t.len()).sum();
    let per_iter = elapsed.as_secs_f32() / iterations as f32;
    println!("  Texts: {}", test_texts.len());
    println!("  Total characters: {}", total_chars);
    println!("  Iterations: {}", iterations);
    println!("  Total time: {:.3}s", elapsed.as_secs_f32());
    println!("  Per iteration: {:.3}ms", per_iter * 1000.0);
    println!(
        "  Throughput: {:.0} chars/sec",
        (total_chars * iterations) as f32 / elapsed.as_secs_f32()
    );
}

fn benchmark_vocoder(iterations: usize) {
    use indextts::vocoder::{create_bigvgan_22k, Vocoder};
    use ndarray::Array2;
    use std::time::Instant;

    println!("\nVocoder Benchmark");
    println!("-----------------");

    let vocoder = create_bigvgan_22k();
    let num_frames = 100; // ~2.5 seconds of audio
    let mel = Array2::zeros((80, num_frames));

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = vocoder.synthesize(&mel);
    }
    let elapsed = start.elapsed();

    let audio_duration = num_frames as f32 * vocoder.hop_length() as f32 / vocoder.sample_rate() as f32;
    let per_iter = elapsed.as_secs_f32() / iterations as f32;
    println!("  Mel frames: {}", num_frames);
    println!("  Audio duration: {:.2}s", audio_duration);
    println!("  Iterations: {}", iterations);
    println!("  Total time: {:.3}s", elapsed.as_secs_f32());
    println!("  Per iteration: {:.3}ms", per_iter * 1000.0);
    println!("  RTF: {:.3}x", per_iter / audio_duration);
}
