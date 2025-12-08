# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IndexTTS-Rust is a high-performance Text-to-Speech engine, a complete Rust rewrite of the Python IndexTTS system. It uses ONNX Runtime for neural network inference and provides zero-shot voice cloning with emotion control.

## Build and Development Commands

```bash
# Build (always build release for performance testing)
cargo build --release

# Run linter (MANDATORY before commits - catches many issues)
cargo clippy -- -D warnings

# Run tests
cargo test

# Run specific test
cargo test test_name

# Run benchmarks (Criterion-based)
cargo bench

# Run specific benchmark
cargo bench --bench mel_spectrogram
cargo bench --bench inference

# Check compilation without building
cargo check

# Format code
cargo fmt

# Full pre-commit workflow (BUILD -> CLIPPY -> BUILD)
cargo build --release && cargo clippy -- -D warnings && cargo build --release
```

## CLI Usage

```bash
# Show help
./target/release/indextts --help

# Synthesize speech
./target/release/indextts synthesize \
  --text "Hello world" \
  --voice examples/voice_01.wav \
  --output output.wav

# Generate default config
./target/release/indextts init-config -o config.yaml

# Show system info
./target/release/indextts info

# Run built-in benchmarks
./target/release/indextts benchmark --iterations 100
```

## Architecture

The codebase follows a modular pipeline architecture where each stage processes data sequentially:

```
Text Input → Normalization → Tokenization → Model Inference → Vocoding → Audio Output
```

### Core Modules (src/)

- **audio/** - Audio DSP operations
  - `mel.rs` - Mel-spectrogram computation (STFT, filterbanks)
  - `io.rs` - WAV file I/O using hound
  - `dsp.rs` - Signal processing utilities
  - `resample.rs` - Audio resampling using rubato

- **text/** - Text processing pipeline
  - `normalizer.rs` - Text normalization (Chinese/English/mixed)
  - `tokenizer.rs` - BPE tokenization via HuggingFace tokenizers
  - `phoneme.rs` - Grapheme-to-phoneme conversion

- **model/** - Neural network inference
  - `session.rs` - ONNX Runtime wrapper (load-dynamic feature)
  - `gpt.rs` - GPT-based sequence generation
  - `embedding.rs` - Speaker and emotion encoders

- **vocoder/** - Neural vocoding
  - `bigvgan.rs` - BigVGAN waveform synthesis
  - `activations.rs` - Snake/SnakeBeta activation functions

- **pipeline/** - TTS orchestration
  - `synthesis.rs` - Main synthesis logic, coordinates all modules

- **config/** - Configuration management (YAML-based via serde)

- **error.rs** - Error types using thiserror

- **lib.rs** - Library entry point, exposes public API

- **main.rs** - CLI entry point using clap

### Key Constants (lib.rs)

```rust
pub const SAMPLE_RATE: u32 = 22050;  // Output audio sample rate
pub const N_MELS: usize = 80;        // Mel filterbank channels
pub const N_FFT: usize = 1024;       // FFT size
pub const HOP_LENGTH: usize = 256;   // STFT hop length
```

### Dependencies Pattern

- **Audio**: hound (WAV), rustfft/realfft (DSP), rubato (resampling), dasp (signal processing)
- **ML Inference**: ort (ONNX Runtime with load-dynamic), ndarray, safetensors
- **Text**: tokenizers (HuggingFace), jieba-rs (Chinese), regex, unicode-segmentation
- **Parallelism**: rayon (data parallelism), tokio (async)
- **CLI**: clap (derive), env_logger, indicatif

## Important Notes

1. **ONNX Runtime**: Uses `load-dynamic` feature - requires ONNX Runtime library installed on system
2. **Model Files**: ONNX models go in `models/` directory (not in git, download separately)
3. **Reference Implementation**: Python code in `indextts - REMOVING - REF ONLY/` is kept for reference only
4. **Performance**: Release builds use LTO and single codegen-unit for maximum optimization
5. **Audio Format**: All internal processing at 22050 Hz, 80-band mel spectrograms

## Testing Strategy

- Unit tests inline in modules
- Criterion benchmarks in `benches/` for performance regression testing
- Python regression tests in `tests/` for end-to-end validation
- Example audio files in `examples/` for testing voice cloning

## Infrastructure

The project includes complete development infrastructure:

- **`scripts/manage.sh`** - Colorful management script with build, test, clean, docker, and more
  ```bash
  ./scripts/manage.sh help          # See all commands
  ./scripts/manage.sh build-full    # Full build workflow (build → clippy → build)
  ./scripts/manage.sh test          # Run all tests
  ./scripts/manage.sh lint          # Run Clippy
  ./scripts/manage.sh docker-build  # Build Docker image
  ./scripts/manage.sh               # Interactive menu
  ```

- **`context.md`** - Conversation continuity for Hue & Aye sessions

- **`tests/`** - Integration tests for ONNX models and full pipeline
  ```bash
  cargo test --test integration_onnx      # ONNX model tests
  cargo test --test integration_pipeline  # Pipeline tests
  ```
