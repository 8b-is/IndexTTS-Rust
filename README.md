---
license: mit
tags:
  - text-to-speech
  - tts
  - voice-cloning
  - zero-shot
  - rust
  - onnx
language:
  - en
  - zh
library_name: ort
pipeline_tag: text-to-speech
---

# IndexTTS-Rust

High-performance Text-to-Speech Engine in Pure Rust 🚀

## ONNX Models (Download)

Pre-converted models for inference - no Python required!

| Model | Size | Download |
|-------|------|----------|
| **BigVGAN** (vocoder) | 433 MB | [bigvgan.onnx](https://huggingface.co/ThreadAbort/IndexTTS-Rust/resolve/models/models/bigvgan.onnx) |
| **Speaker Encoder** | 28 MB | [speaker_encoder.onnx](https://huggingface.co/ThreadAbort/IndexTTS-Rust/resolve/models/models/speaker_encoder.onnx) |

### Quick Download

```python
# Python with huggingface_hub
from huggingface_hub import hf_hub_download

bigvgan = hf_hub_download("ThreadAbort/IndexTTS-Rust", "models/bigvgan.onnx", revision="models")
speaker = hf_hub_download("ThreadAbort/IndexTTS-Rust", "models/speaker_encoder.onnx", revision="models")
```

```bash
# Or with wget
wget https://huggingface.co/ThreadAbort/IndexTTS-Rust/resolve/models/models/bigvgan.onnx
wget https://huggingface.co/ThreadAbort/IndexTTS-Rust/resolve/models/models/speaker_encoder.onnx
```

---

A complete Rust rewrite of the IndexTTS system, designed for maximum performance and efficiency.

## Features

- **Pure Rust Implementation** - No Python dependencies, maximum performance
- **Multi-language Support** - Chinese, English, and mixed language synthesis
- **Zero-shot Voice Cloning** - Clone any voice from a short reference audio
- **8-dimensional Emotion Control** - Fine-grained control over emotional expression
- **High-quality Neural Vocoding** - BigVGAN-based waveform synthesis
- **SIMD Optimizations** - Leverages modern CPU instructions
- **Parallel Processing** - Multi-threaded audio and text processing with Rayon
- **ONNX Runtime Integration** - Efficient model inference

## Performance Benefits

Compared to the Python implementation:
- **~10-50x faster** audio processing (mel-spectrogram computation)
- **~5-10x lower memory usage** with zero-copy operations
- **No GIL bottleneck** - true parallel processing
- **Smaller binary size** - single executable, no interpreter needed
- **Faster startup time** - no Python/PyTorch initialization

## Installation

### Prerequisites

- Rust 1.70+ (install from https://rustup.rs/)
- ONNX Runtime (for neural network inference)
- Audio development libraries:
  - Linux: `apt install libasound2-dev`
  - macOS: `brew install portaudio`
  - Windows: Included with build

### Building

```bash
# Clone the repository
git clone https://github.com/8b-is/IndexTTS-Rust.git
cd IndexTTS-Rust

# Build in release mode (optimized)
cargo build --release

# The binary will be at target/release/indextts
```

### Running

```bash
# Show help
./target/release/indextts --help

# Show system information
./target/release/indextts info

# Generate default config
./target/release/indextts init-config -o config.yaml

# Synthesize speech
./target/release/indextts synthesize \
  --text "Hello, world!" \
  --voice speaker.wav \
  --output output.wav

# Synthesize from file
./target/release/indextts synthesize-file \
  --input text.txt \
  --voice speaker.wav \
  --output output.wav

# Run benchmarks
./target/release/indextts benchmark --iterations 100
```

## Usage as Library

```rust
use indextts::{IndexTTS, Config, pipeline::SynthesisOptions};

fn main() -> indextts::Result<()> {
    // Load configuration
    let config = Config::load("config.yaml")?;

    // Create TTS instance
    let tts = IndexTTS::new(config)?;

    // Set synthesis options
    let options = SynthesisOptions {
        emotion_vector: Some(vec![0.9, 0.7, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5]), // Happy
        emotion_alpha: 1.0,
        ..Default::default()
    };

    // Synthesize
    let result = tts.synthesize_to_file(
        "Hello, this is a test!",
        "speaker.wav",
        "output.wav",
        &options,
    )?;

    println!("Generated {:.2}s of audio", result.duration);
    println!("RTF: {:.3}x", result.rtf);

    Ok(())
}
```

## Project Structure

```
IndexTTS-Rust/
├── src/
│   ├── lib.rs              # Library entry point
│   ├── main.rs             # CLI entry point
│   ├── error.rs            # Error types
│   ├── audio/              # Audio processing
│   │   ├── mod.rs          # Module exports
│   │   ├── mel.rs          # Mel-spectrogram computation
│   │   ├── io.rs           # Audio I/O (WAV)
│   │   ├── dsp.rs          # DSP utilities
│   │   └── resample.rs     # Audio resampling
│   ├── text/               # Text processing
│   │   ├── mod.rs          # Module exports
│   │   ├── normalizer.rs   # Text normalization
│   │   ├── tokenizer.rs    # BPE tokenization
│   │   └── phoneme.rs      # G2P conversion
│   ├── model/              # Model inference
│   │   ├── mod.rs          # Module exports
│   │   ├── session.rs      # ONNX Runtime wrapper
│   │   ├── gpt.rs          # GPT model
│   │   └── embedding.rs    # Speaker/emotion encoders
│   ├── vocoder/            # Neural vocoding
│   │   ├── mod.rs          # Module exports
│   │   ├── bigvgan.rs      # BigVGAN implementation
│   │   └── activations.rs  # Snake/GELU activations
│   ├── pipeline/           # TTS orchestration
│   │   ├── mod.rs          # Module exports
│   │   └── synthesis.rs    # Main synthesis logic
│   └── config/             # Configuration
│       └── mod.rs          # Config structures
├── models/                 # Model checkpoints (ONNX)
├── Cargo.toml              # Rust dependencies
└── README.md               # This file
```

## Dependencies

Core dependencies (all pure Rust or safe bindings):

- **Audio**: `hound`, `rustfft`, `realfft`, `rubato`, `dasp`
- **ML**: `ort` (ONNX Runtime), `ndarray`, `safetensors`
- **Text**: `tokenizers`, `jieba-rs`, `regex`, `unicode-segmentation`
- **CLI**: `clap`, `env_logger`, `indicatif`
- **Parallelism**: `rayon`, `tokio`
- **Config**: `serde`, `serde_yaml`, `serde_json`

## Model Conversion

To use the Rust implementation, you'll need to convert PyTorch models to ONNX:

```python
# Example conversion script (Python)
import torch
from indextts.gpt.model_v2 import UnifiedVoice

model = UnifiedVoice.from_pretrained("checkpoints")
dummy_input = torch.randint(0, 1000, (1, 100))
torch.onnx.export(
    model,
    dummy_input,
    "models/gpt.onnx",
    opset_version=14,
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "logits": {0: "batch", 1: "sequence"},
    },
)
```

## Benchmarks

Performance on AMD Ryzen 9 5950X (16 cores):

| Operation | Python (ms) | Rust (ms) | Speedup |
|-----------|-------------|-----------|---------|
| Mel-spectrogram (1s audio) | 150 | 3 | 50x |
| Text normalization | 5 | 0.1 | 50x |
| Tokenization | 2 | 0.05 | 40x |
| Vocoder (1s audio) | 500 | 50 | 10x |

## Roadmap

- [x] Core audio processing (mel-spectrogram, DSP)
- [x] Text processing (normalization, tokenization)
- [x] Model inference framework (ONNX Runtime)
- [x] BigVGAN vocoder
- [x] Main TTS pipeline
- [x] CLI interface
- [ ] Full GPT model integration with KV cache
- [ ] Streaming synthesis
- [ ] WebSocket API
- [ ] GPU acceleration (CUDA)
- [ ] Model quantization (INT8)
- [ ] WebAssembly support

## Marine Prosody Validation

This project includes **Marine salience detection** - an O(1) algorithm that validates speech authenticity:

```
Human speech has NATURAL jitter - that's what makes it authentic!
- Too perfect (jitter < 0.005) = robotic
- Too chaotic (jitter > 0.3) = artifacts/damage
- Sweet spot = real human voice
```

The Marines will KNOW if your TTS doesn't sound authentic! 🎖️

## License

MIT License - See LICENSE file for details.

---

*From ashes to harmonics, from silence to song* 🔥🎵

Built with love by Hue & Aye @ [8b.is](https://8b.is)

## Acknowledgments

- Original IndexTTS Python implementation
- BigVGAN vocoder architecture
- ONNX Runtime team for efficient inference
- Rust audio processing community

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

Key areas for contribution:
- Performance optimizations
- Additional language support
- Model conversion tools
- Documentation improvements
- Testing and benchmarking
