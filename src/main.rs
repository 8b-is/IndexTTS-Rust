//! IndexTTS CLI - High-performance Text-to-Speech in Rust
//!
//! Command-line interface for IndexTTS synthesizer.
//! Built with love by Hue & Aye @ 8b.is 💜
//!
//! # Quick Start
//! ```bash
//! # Check system status
//! indextts diagnose
//!
//! # Clone a voice
//! indextts clone -t "Hello world" -v examples/chris.wav -o output.wav
//!
//! # Run demo
//! indextts demo
//! ```

use clap::{Parser, Subcommand};
use indextts::{
    audio::{load_audio, save_audio, AudioData},
    model::{check_ort_availability, ModelCache, OrtStatus},
    pipeline::{IndexTTS, SynthesisOptions},
    quality::MarineProsodyConditioner,
    Config, Result, SAMPLE_RATE,
};
use std::path::PathBuf;

// ============================================================================
// CLI Structure
// ============================================================================

#[derive(Parser)]
#[command(
    name = "indextts",
    about = "🎤 IndexTTS - High-performance Text-to-Speech in Pure Rust 🦀",
    version,
    author = "Hue & Aye @ 8b.is",
    after_help = "Examples:\n  indextts clone -t \"Hello\" -v voice.wav\n  indextts diagnose\n  indextts demo"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// 🎙️ Clone a voice and synthesize speech
    Clone {
        /// Text to synthesize
        #[arg(short, long)]
        text: String,

        /// Speaker reference audio file (WAV)
        #[arg(short = 'v', long)]
        voice: PathBuf,

        /// Output audio file path
        #[arg(short, long, default_value = "output.wav")]
        output: PathBuf,

        /// Model directory (default: models, or checkpoints)
        #[arg(short, long)]
        model_dir: Option<PathBuf>,

        /// Configuration file (optional)
        #[arg(short, long)]
        config: Option<PathBuf>,

        /// Emotion vector (8 comma-separated floats 0-1)
        #[arg(long)]
        emotion: Option<String>,

        /// Emotion strength (0-1)
        #[arg(long, default_value = "1.0")]
        emotion_alpha: f32,
    },

    /// 🔍 Diagnose system and model status
    Diagnose {
        /// Model directory to check
        #[arg(short, long, default_value = "models")]
        model_dir: PathBuf,
    },

    /// 🎵 Run a demo with sample audio
    Demo {
        /// Output audio file
        #[arg(short, long, default_value = "demo_output.wav")]
        output: PathBuf,
    },

    /// 📊 Analyze voice prosody using Marine algorithm
    Analyze {
        /// Audio file to analyze
        #[arg(short, long)]
        audio: PathBuf,
    },

    /// ⚙️ Generate default configuration file
    InitConfig {
        /// Output path for config file
        #[arg(short, long, default_value = "config.yaml")]
        output: PathBuf,
    },

    /// ℹ️ Show system information
    Info,

    /// 📈 Run performance benchmarks
    Benchmark {
        /// Number of iterations
        #[arg(short, long, default_value = "10")]
        iterations: usize,
    },

    /// 🌐 Start TTS TCP server (Hot Tub listener)
    Serve {
        /// Port to listen on
        #[arg(short, long, default_value = "8420")]
        port: u16,

        /// Model directory
        #[arg(short, long)]
        model_dir: Option<PathBuf>,

        /// Configuration file
        #[arg(short, long)]
        config: Option<PathBuf>,
    },

    // Legacy alias for synthesize
    /// Synthesize speech (alias for clone)
    #[command(hide = true)]
    Synthesize {
        #[arg(short, long)]
        text: String,
        #[arg(short = 'v', long)]
        voice: PathBuf,
        #[arg(short, long, default_value = "output.wav")]
        output: PathBuf,
        #[arg(short, long)]
        config: Option<PathBuf>,
        #[arg(short, long, default_value = "models")]
        model_dir: PathBuf,
        #[arg(long)]
        emotion: Option<String>,
        #[arg(long, default_value = "1.0")]
        emotion_alpha: f32,
        #[arg(long, default_value = "50")]
        top_k: usize,
        #[arg(long, default_value = "0.95")]
        top_p: f32,
        #[arg(long, default_value = "1.1")]
        repetition_penalty: f32,
        #[arg(long)]
        fp16: bool,
        #[arg(short, long, default_value = "cpu")]
        device: String,
    },
}

// ============================================================================
// Main Entry Point
// ============================================================================

fn main() -> Result<()> {
    // Initialize logger with nice formatting
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp(None)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Clone {
            text,
            voice,
            output,
            model_dir,
            config,
            emotion,
            emotion_alpha,
        } => cmd_clone(text, voice, output, model_dir, config, emotion, emotion_alpha),

        Commands::Diagnose { model_dir } => cmd_diagnose(model_dir),

        Commands::Demo { output } => cmd_demo(output),

        Commands::Analyze { audio } => cmd_analyze(audio),

        Commands::InitConfig { output } => cmd_init_config(output),

        Commands::Info => cmd_info(),

        Commands::Benchmark { iterations } => cmd_benchmark(iterations),

        Commands::Serve { port, model_dir, config } => cmd_serve(port, model_dir, config),

        Commands::Synthesize {
            text,
            voice,
            output,
            config,
            model_dir,
            emotion,
            emotion_alpha,
            ..
        } => cmd_clone(text, voice, output, Some(model_dir), config, emotion, emotion_alpha),
    }
}

// ============================================================================
// Command Implementations
// ============================================================================

/// Clone a voice and synthesize speech
fn cmd_clone(
    text: String,
    voice: PathBuf,
    output: PathBuf,
    model_dir: Option<PathBuf>,
    config: Option<PathBuf>,
    emotion: Option<String>,
    emotion_alpha: f32,
) -> Result<()> {
    println!();
    println!("🎤 IndexTTS Voice Cloning");
    println!("========================");
    println!();

    // Determine model directory
    let model_dir = model_dir.unwrap_or_else(|| {
        if PathBuf::from("checkpoints").exists() {
            PathBuf::from("checkpoints")
        } else {
            PathBuf::from("models")
        }
    });

    // Check voice file exists
    if !voice.exists() {
        println!("❌ Voice file not found: {}", voice.display());
        return Err(indextts::Error::FileNotFound(voice.display().to_string()));
    }

    println!("📝 Text: \"{}\"", &text[..text.len().min(80)]);
    println!("🎙️ Voice: {}", voice.display());
    println!("📂 Models: {}", model_dir.display());
    println!("💾 Output: {}", output.display());
    println!();

    // Check model status
    let cache = ModelCache::new(&model_dir);
    let required_models = ["gpt", "s2mel", "bigvgan", "speaker_encoder"];
    let (_available, missing) = cache.check_required_models(&required_models);

    if !missing.is_empty() {
        println!("⚠️  Missing ONNX models: {:?}", missing);
        println!();
        println!("The following models need to be converted to ONNX format:");
        for model in &missing {
            let pth_path = model_dir.join(format!("{}.pth", model));
            if pth_path.exists() {
                println!("   {} - PyTorch file exists, needs ONNX export", model);
            } else {
                println!("   {} - Not found", model);
            }
        }
        println!();

        // Check if we have the critical GPT model
        if missing.contains(&"gpt".to_string()) || missing.contains(&"s2mel".to_string()) {
            println!("🔧 To convert PyTorch models to ONNX, you can use:");
            println!("   python scripts/export_onnx.py --model gpt --output models/gpt.onnx");
            println!();
            println!("For now, running in DEMO MODE (placeholder synthesis)...");
            println!();
        }
    }

    // Load or create config
    let cfg = if let Some(config_path) = config {
        Config::load(config_path)?
    } else {
        // Try to load config from checkpoints if it exists
        let checkpoint_config = model_dir.join("config.yaml");
        if checkpoint_config.exists() {
            println!("📋 Loading config from: {}", checkpoint_config.display());
            let mut cfg = Config::load(&checkpoint_config).unwrap_or_default();
            cfg.model_dir = model_dir.clone();
            cfg
        } else {
            Config {
                model_dir: model_dir.clone(),
                ..Default::default()
            }
        }
    };

    // Create TTS instance
    println!("🚀 Initializing IndexTTS...");
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
        ..Default::default()
    };

    // Synthesize
    println!("🎵 Synthesizing speech...");
    let result = tts.synthesize_to_file(
        &text,
        voice.to_str().unwrap(),
        output.to_str().unwrap(),
        &options,
    )?;

    println!();
    println!("✅ Synthesis complete!");
    println!("   Duration: {}", result.duration_formatted());
    println!("   Processing: {:.2}s", result.processing_time);
    println!("   RTF: {:.3}x", result.rtf);
    println!();
    println!("🎧 Output saved to: {}", output.display());
    println!();

    // Run Marine analysis on output
    println!("🌊 Running Marine quality analysis...");
    if let Ok(audio) = load_audio(output.to_str().unwrap(), Some(SAMPLE_RATE)) {
        let conditioner = MarineProsodyConditioner::new(SAMPLE_RATE);
        if let Ok(report) = conditioner.validate_tts_output(&audio.samples) {
            println!("   Quality Score: {:.1}%", report.quality_score);
            if report.issues.is_empty() {
                println!("   Status: ✅ No issues detected");
            } else {
                for issue in &report.issues {
                    println!("   ⚠️  {}", issue);
                }
            }
        }
    }

    Ok(())
}

/// Diagnose system status
fn cmd_diagnose(model_dir: PathBuf) -> Result<()> {
    println!();
    println!("🔍 IndexTTS System Diagnostics");
    println!("==============================");
    println!();

    // Check ONNX Runtime
    println!("ONNX Runtime:");
    match check_ort_availability() {
        OrtStatus::Available => {
            println!("   ✅ Available");
            if let Ok(path) = std::env::var("ORT_DYLIB_PATH") {
                println!("   📍 Path: {}", path);
            }
        }
        OrtStatus::LibraryNotFound => {
            println!("   ❌ Not found");
            println!("   💡 Set ORT_DYLIB_PATH to the path of libonnxruntime.dylib");
            println!("      export ORT_DYLIB_PATH=/path/to/libonnxruntime.dylib");
        }
        OrtStatus::InitFailed(e) => {
            println!("   ⚠️  Init failed: {}", e);
        }
    }
    println!();

    // Check model directories
    println!("Model Directories:");
    for dir in &["models", "checkpoints"] {
        let path = PathBuf::from(dir);
        if path.exists() {
            print!("   ✅ {}/", dir);
            // Count files
            if let Ok(entries) = std::fs::read_dir(&path) {
                let count = entries.count();
                println!(" ({} files)", count);
            } else {
                println!();
            }
        } else {
            println!("   ❌ {}/ (not found)", dir);
        }
    }
    println!();

    // Check ONNX models
    println!("ONNX Models (in {}):", model_dir.display());
    let onnx_models = ["gpt", "s2mel", "bigvgan", "speaker_encoder"];
    for model in &onnx_models {
        let path = model_dir.join(format!("{}.onnx", model));
        if path.exists() {
            let size = std::fs::metadata(&path)
                .map(|m| format!("{:.1} MB", m.len() as f64 / 1_048_576.0))
                .unwrap_or_else(|_| "?".to_string());
            println!("   ✅ {}.onnx ({})", model, size);
        } else {
            println!("   ❌ {}.onnx", model);
        }
    }
    println!();

    // Check PyTorch models (in checkpoints)
    if PathBuf::from("checkpoints").exists() {
        println!("PyTorch Models (in checkpoints/):");
        let pth_models = ["gpt", "s2mel"];
        for model in &pth_models {
            let path = PathBuf::from("checkpoints").join(format!("{}.pth", model));
            if path.exists() {
                let size = std::fs::metadata(&path)
                    .map(|m| format!("{:.1} GB", m.len() as f64 / 1_073_741_824.0))
                    .unwrap_or_else(|_| "?".to_string());
                println!("   📦 {}.pth ({}) - needs ONNX export", model, size);
            }
        }

        // Check for tokenizer
        let bpe_path = PathBuf::from("checkpoints/bpe.model");
        if bpe_path.exists() {
            println!("   ✅ bpe.model (tokenizer)");
        }

        // Check for config
        let config_path = PathBuf::from("checkpoints/config.yaml");
        if config_path.exists() {
            println!("   ✅ config.yaml");
        }
        println!();
    }

    // Check example files
    if PathBuf::from("examples").exists() {
        println!("Example Audio Files:");
        if let Ok(entries) = std::fs::read_dir("examples") {
            let wavs: Vec<_> = entries
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().is_some_and(|ext| ext == "wav"))
                .take(5)
                .collect();

            for entry in &wavs {
                let name = entry.file_name();
                let size = std::fs::metadata(entry.path())
                    .map(|m| format!("{:.1} MB", m.len() as f64 / 1_048_576.0))
                    .unwrap_or_else(|_| "?".to_string());
                println!("   🎵 {} ({})", name.to_string_lossy(), size);
            }
            if wavs.len() == 5 {
                println!("   ... and more");
            }
        }
        println!();
    }

    // Summary
    println!("📋 Summary:");
    let cache = ModelCache::new(&model_dir);
    let (_available, missing) = cache.check_required_models(&onnx_models);

    if missing.is_empty() {
        println!("   ✅ All required ONNX models present");
        println!("   🚀 Ready for voice cloning!");
    } else {
        println!("   ⚠️  Missing {} of {} required models", missing.len(), onnx_models.len());
        println!();
        println!("To convert PyTorch models to ONNX:");
        println!("   1. Install onnx: pip install onnx onnxruntime");
        println!("   2. Run export script (or use Python to export)");
        println!();
        println!("For now, you can still run demos and test the pipeline.");
    }

    println!();
    Ok(())
}

/// Run a demo
fn cmd_demo(output: PathBuf) -> Result<()> {
    println!();
    println!("🎵 IndexTTS Demo Mode");
    println!("====================");
    println!();

    // Generate a simple demo audio
    println!("Generating demo audio...");

    // Create a simple 440Hz tone for 2 seconds
    let duration_secs = 2.0;
    let num_samples = (SAMPLE_RATE as f32 * duration_secs) as usize;

    let samples: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE as f32;
            // Create a pleasant chord with fade
            let fade = if t < 0.1 {
                t / 0.1
            } else if t > duration_secs - 0.1 {
                (duration_secs - t) / 0.1
            } else {
                1.0
            };

            let f1 = (2.0 * std::f32::consts::PI * 261.63 * t).sin(); // C4
            let f2 = (2.0 * std::f32::consts::PI * 329.63 * t).sin(); // E4
            let f3 = (2.0 * std::f32::consts::PI * 392.00 * t).sin(); // G4

            (f1 + f2 * 0.7 + f3 * 0.5) * 0.3 * fade
        })
        .collect();

    let audio_data = AudioData::new(samples, SAMPLE_RATE);
    save_audio(output.to_str().unwrap(), &audio_data)?;

    println!("✅ Demo audio saved to: {}", output.display());
    println!();
    println!("This is a placeholder demo. For real voice cloning:");
    println!("   1. Run: indextts diagnose");
    println!("   2. Ensure ONNX models are available");
    println!("   3. Run: indextts clone -t \"Your text\" -v voice.wav");

    Ok(())
}

/// Analyze audio prosody
fn cmd_analyze(audio: PathBuf) -> Result<()> {
    println!();
    println!("🌊 Marine Prosody Analysis");
    println!("=========================");
    println!();

    if !audio.exists() {
        println!("❌ Audio file not found: {}", audio.display());
        return Err(indextts::Error::FileNotFound(audio.display().to_string()));
    }

    println!("📂 Analyzing: {}", audio.display());
    println!();

    // Load audio
    let audio_data = load_audio(audio.to_str().unwrap(), Some(SAMPLE_RATE))?;
    let duration = audio_data.samples.len() as f32 / SAMPLE_RATE as f32;

    println!("Audio Info:");
    println!("   Samples: {}", audio_data.samples.len());
    println!("   Duration: {:.2}s", duration);
    println!("   Sample Rate: {} Hz", audio_data.sample_rate);
    println!();

    // Extract prosody
    let conditioner = MarineProsodyConditioner::new(SAMPLE_RATE);

    match conditioner.from_samples(&audio_data.samples) {
        Ok(prosody) => {
            println!("Marine Prosody Vector (8D):");
            println!("   jp_mean (pitch jitter):    {:.4}", prosody.jp_mean);
            println!("   jp_std (pitch variance):   {:.4}", prosody.jp_std);
            println!("   ja_mean (amp jitter):      {:.4}", prosody.ja_mean);
            println!("   ja_std (amp variance):     {:.4}", prosody.ja_std);
            println!("   h_mean (harmonic):         {:.4}", prosody.h_mean);
            println!("   s_mean (salience):         {:.4}", prosody.s_mean);
            println!("   peak_density:              {:.2}", prosody.peak_density);
            println!("   energy_mean:               {:.4}", prosody.energy_mean);
            println!();

            // Interpret
            println!("Interpretation:");

            // Estimate emotional state
            let valence = prosody.estimate_valence();
            let arousal = prosody.estimate_arousal();

            if valence > 0.3 {
                println!("   😊 Positive valence (confident, happy)");
            } else if valence < -0.3 {
                println!("   🤔 Negative valence (processing, thoughtful)");
            } else {
                println!("   😐 Neutral valence");
            }

            if arousal > 0.6 {
                println!("   ⚡ High arousal (energetic, expressive)");
            } else if arousal < 0.4 {
                println!("   😌 Low arousal (calm, relaxed)");
            } else {
                println!("   📊 Moderate arousal");
            }

            // Quality assessment
            if prosody.jp_mean < 0.005 {
                println!("   ⚠️  Very low jitter - may sound robotic");
            } else if prosody.jp_mean > 0.3 {
                println!("   ⚠️  High jitter - may indicate artifacts");
            } else {
                println!("   ✅ Natural jitter range");
            }
        }
        Err(e) => {
            println!("❌ Analysis failed: {}", e);
        }
    }

    println!();
    Ok(())
}

/// Generate default config
fn cmd_init_config(output: PathBuf) -> Result<()> {
    println!("⚙️ Creating default configuration...");

    let config = Config::default();
    config.save(&output)?;

    println!("✅ Configuration saved to: {}", output.display());
    Ok(())
}

/// Show system info
fn cmd_info() -> Result<()> {
    println!();
    println!("🎤 IndexTTS - High-performance Text-to-Speech");
    println!("=============================================");
    println!();
    println!("Version:       {}", indextts::VERSION);
    println!("Platform:      {} / {}", std::env::consts::OS, std::env::consts::ARCH);
    println!("CPU Cores:     {} ({} physical)", num_cpus::get(), num_cpus::get_physical());
    println!();
    println!("Audio Settings:");
    println!("   Sample Rate:  {} Hz", SAMPLE_RATE);
    println!("   Mel Bands:    {}", indextts::N_MELS);
    println!("   FFT Size:     {}", indextts::N_FFT);
    println!("   Hop Length:   {}", indextts::HOP_LENGTH);
    println!();
    println!("Features:");
    println!("   ✅ Multi-language (English, Chinese, mixed)");
    println!("   ✅ Zero-shot voice cloning");
    println!("   ✅ 8D emotion control (Marine prosody)");
    println!("   ✅ BigVGAN neural vocoding");
    println!("   ✅ SIMD-optimized audio processing");
    println!();
    println!("Built with 💜 by Hue & Aye @ 8b.is");
    println!();

    Ok(())
}

/// Run benchmarks
fn cmd_benchmark(iterations: usize) -> Result<()> {
    println!();
    println!("📈 IndexTTS Benchmarks");
    println!("=====================");
    println!("Iterations: {}", iterations);
    println!();

    // Benchmark mel-spectrogram
    benchmark_mel_spectrogram(iterations);

    // Benchmark tokenization
    benchmark_tokenization(iterations);

    // Benchmark vocoder
    benchmark_vocoder(iterations);

    println!("✅ Benchmarks complete");
    Ok(())
}

// ============================================================================
// Benchmark Helpers
// ============================================================================

fn benchmark_mel_spectrogram(iterations: usize) {
    use indextts::audio::{mel_spectrogram, AudioConfig};
    use std::time::Instant;

    println!("Mel-Spectrogram:");

    let config = AudioConfig::default();
    let num_samples = config.sample_rate as usize; // 1 second
    let signal: Vec<f32> = (0..num_samples)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = mel_spectrogram(&signal, &config);
    }
    let elapsed = start.elapsed();

    let per_iter = elapsed.as_secs_f32() / iterations as f32;
    println!("   Signal: {} samples ({:.2}s)", num_samples, num_samples as f32 / config.sample_rate as f32);
    println!("   Per iter: {:.3}ms", per_iter * 1000.0);
    println!("   RTF: {:.1}x", 1.0 / per_iter);
    println!();
}

fn benchmark_tokenization(iterations: usize) {
    use indextts::text::{TextNormalizer, TextTokenizer, TokenizerConfig};
    use std::time::Instant;

    println!("Tokenization:");

    let normalizer = TextNormalizer::new();
    let tokenizer = match TextTokenizer::new(TokenizerConfig::default()) {
        Ok(t) => t,
        Err(e) => {
            println!("   ⚠️ Could not init tokenizer: {}", e);
            return;
        }
    };

    let texts = vec![
        "Hello world, this is a test.",
        "The quick brown fox jumps over the lazy dog.",
        "你好世界，这是一个测试。",
        "Mixed: Hello 世界 and 你好 world.",
    ];

    let start = Instant::now();
    for _ in 0..iterations {
        for text in &texts {
            if let Ok(norm) = normalizer.normalize(text) {
                let _ = tokenizer.encode(&norm);
            }
        }
    }
    let elapsed = start.elapsed();

    let total_chars: usize = texts.iter().map(|t| t.len()).sum();
    let per_iter = elapsed.as_secs_f32() / iterations as f32;
    println!("   Texts: {} ({} chars)", texts.len(), total_chars);
    println!("   Per iter: {:.3}ms", per_iter * 1000.0);
    println!("   Throughput: {:.0} chars/sec", (total_chars * iterations) as f32 / elapsed.as_secs_f32());
    println!();
}

fn benchmark_vocoder(iterations: usize) {
    use indextts::vocoder::{create_bigvgan_22k, Vocoder};
    use ndarray::Array2;
    use std::time::Instant;

    println!("Vocoder (BigVGAN):");

    let vocoder = create_bigvgan_22k();
    let num_frames = 100; // ~2.5 seconds
    let mel = Array2::zeros((80, num_frames));

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = vocoder.synthesize(&mel);
    }
    let elapsed = start.elapsed();

    let audio_duration = num_frames as f32 * vocoder.hop_length() as f32 / vocoder.sample_rate() as f32;
    let per_iter = elapsed.as_secs_f32() / iterations as f32;
    println!("   Mel frames: {}", num_frames);
    println!("   Audio: {:.2}s", audio_duration);
    println!("   Per iter: {:.3}ms", per_iter * 1000.0);
    println!("   RTF: {:.3}x", per_iter / audio_duration);
    println!();
}

// ============================================================================
// Server Implementation
// ============================================================================

use serde::Deserialize;

#[derive(Deserialize, Debug)]
struct SpeechMessage {
    voice: String,
    text: String,
    priority: String,
}

#[tokio::main]
async fn cmd_serve(port: u16, model_dir: Option<PathBuf>, config: Option<PathBuf>) -> Result<()> {
    println!();
    println!("🌐 IndexTTS Hot Tub Server");
    println!("========================");
    println!();

    // Determine model directory
    let model_dir = model_dir.unwrap_or_else(|| {
        if PathBuf::from("checkpoints").exists() {
            PathBuf::from("checkpoints")
        } else {
            PathBuf::from("models")
        }
    });

    // Load or create config
    let cfg = if let Some(config_path) = config {
        Config::load(config_path)?
    } else {
        let checkpoint_config = model_dir.join("config.yaml");
        if checkpoint_config.exists() {
            println!("📋 Loading config from: {}", checkpoint_config.display());
            let mut cfg = Config::load(&checkpoint_config).unwrap_or_default();
            cfg.model_dir = model_dir.clone();
            cfg
        } else {
            Config {
                model_dir: model_dir.clone(),
                ..Default::default()
            }
        }
    };

    println!("🚀 Initializing IndexTTS Engine...");
    let tts = std::sync::Arc::new(IndexTTS::new(cfg)?);

    let addr = format!("127.0.0.1:{}", port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    println!("✅ Listening on TCP {}", addr);
    println!();

    loop {
        let (mut socket, peer) = listener.accept().await?;
        println!("📡 Accepted connection from {}", peer);
        
        let tts_clone = std::sync::Arc::clone(&tts);
        
        tokio::spawn(async move {
            use tokio::io::AsyncReadExt;
            let mut buf = vec![0; 4096];
            loop {
                match socket.read(&mut buf).await {
                    Ok(0) => break,
                    Ok(n) => {
                        let json_str = String::from_utf8_lossy(&buf[..n]);
                        for line in json_str.lines() {
                            if line.trim().is_empty() { continue; }
                            match serde_json::from_str::<SpeechMessage>(line) {
                                Ok(msg) => {
                                    println!("🗣️  Request [{}]: \"{}\"", msg.voice, msg.text);
                                    let options = SynthesisOptions::default();
                                    // Use a dummy speaker audio path or mapping. We will use a fallback or demo if not found.
                                    // In a real implementation we would look up the voice reference audio.
                                    let voice_path = format!("examples/{}.wav", msg.voice.to_lowercase());
                                    let safe_voice_path = if PathBuf::from(&voice_path).exists() {
                                        voice_path
                                    } else {
                                        "examples/default.wav".to_string()
                                    };
                                    
                                    // Make sure examples/default.wav exists or ignore the error if it's missing for demo mode
                                    if let Err(e) = tts_clone.synthesize(&msg.text, &safe_voice_path, &options) {
                                        println!("⚠️  Synthesis error: {}", e);
                                    } else {
                                        println!("✅ Synthesized successfully");
                                    }
                                }
                                Err(e) => {
                                    println!("❌ JSON parsing error: {}", e);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        println!("❌ Socket read error: {}", e);
                        break;
                    }
                }
            }
            println!("🔌 Disconnected {}", peer);
        });
    }
}
