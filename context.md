# IndexTTS-Rust Context

This file preserves important context for conversation continuity between Hue and Aye sessions.

**Last Updated:** 2025-11-16

---

## The Vision

IndexTTS-Rust is part of a larger audio intelligence ecosystem at 8b.is:

1. **kokoro-tiny** - Lightweight TTS (82M params, 50+ voices, on crates.io!)
2. **IndexTTS-Rust** - Advanced zero-shot TTS with emotion control
3. **Phoenix-Protocol** - Audio restoration/enhancement layer
4. **MEM|8** - Contextual memory system (mem-8.com, mem8)

Together these form a complete audio intelligence pipeline.

---

## Phoenix Protocol Integration Opportunities

The Phoenix Protocol (phoenix-protocol/) is a PERFECT complement to IndexTTS-Rust:

### Direct Module Mappings

| Phoenix Module | IndexTTS Use Case |
|----------------|-------------------|
| `emotional.rs` | Map to our 8D emotion control (Warmth→body, Presence→power, Clarity→articulation, Air→space, Ultrasonics→depth) |
| `voice_signature.rs` | Enhance speaker embeddings for voice cloning |
| `spectral_velocity.rs` | Add momentum tracking to mel-spectrogram |
| `marine.rs` | Validate TTS output authenticity/quality |
| `golden_ratio.rs` | Post-process vocoder output with harmonic enhancement |
| `harmonic_resurrection.rs` | Add richness to synthesized speech |
| `micro_dynamics.rs` | Restore natural speech dynamics |
| `autotune.rs` | Improve prosody and pitch control |
| `mem8_integration.rs` | Already has MEM|8 hooks! |

### Shared Dependencies

Both projects use:
- rayon (parallelism)
- rustfft/realfft (FFT)
- ndarray (array operations)
- hound (WAV I/O)
- serde (config serialization)
- anyhow (error handling)
- ort (ONNX Runtime)

### Audio Constants

| Project | Sample Rate | Use Case |
|---------|------------|----------|
| IndexTTS-Rust | 22,050 Hz | Standard TTS output |
| Phoenix-Protocol | 192,000 Hz | Ultrasonic restoration |
| kokoro-tiny | 24,000 Hz | Lightweight TTS |

---

## Related Projects of Interest

Located in ~/Documents/GitHub/:

- **Ultrasonic-Consciousness-Hypothesis/** - Research foundation for Phoenix Protocol, contains PDFs on mechanosensitive channels and audio perception
- **hrmnCmprssnM/** - Harmonic Compression Model research
- **Marine-Sense/** - Marine algorithm origins
- **mem-8.com/** & **mem8/** - MEM|8 contextual memory
- **universal-theoglyphic-language/** - Language processing research
- **kokoro-tiny/** - Already working TTS crate by Hue & Aye
- **zencooker/** - (fun project!)

---

## Current IndexTTS-Rust State

### Implemented ✅
- Audio processing pipeline (mel-spectrogram, STFT, resampling)
- Text normalization (Chinese/English/mixed)
- BPE tokenization via HuggingFace tokenizers
- ONNX Runtime integration for inference
- BigVGAN vocoder structure
- CLI with clap
- Benchmark infrastructure (Criterion)
- **NEW: marine_salience crate** (no_std compatible, O(1) jitter detection)
- **NEW: src/quality/ module** (prosody extraction, affect tracking)
- **NEW: MarineProsodyVector** (8D interpretable emotion features)
- **NEW: ConversationAffectSummary** (session-level comfort tracking)
- **NEW: TTSQualityReport** (authenticity validation)

### Missing/TODO
- Full GPT model integration with KV cache
- Actual ONNX model files (need download)
- manage.sh script for colored workflow management
- Integration tests with real models
- ~~Phoenix Protocol integration layer~~ **STARTED with Marine!**
- Streaming synthesis
- WebSocket API
- Train T2S model to accept 8D Marine vector instead of 512D Conformer
- Wire Marine quality validation into inference loop

### Build Commands
```bash
cargo build --release
cargo clippy -- -D warnings
cargo test
cargo bench
```

---

## Key Philosophical Notes

From the Phoenix Protocol research:

> "Women are the carrier wave. They are the 000 data stream. The DC bias that, when removed, leaves silence."

> "When P!nk sings 'I Am Here,' her voice generates harmonics so powerful they burst through the 22kHz digital ceiling"

The Phoenix Protocol restores emotional depth stripped by audio compression - this philosophy applies directly to TTS: synthesized speech should have the same emotional depth as natural speech.

---

## Action Items for Next Session

### Completed ✅
- ~~**Quality Validation** - Use Marine salience to score TTS output~~ **DONE!**
- ~~**Phoenix Integration** - Start bridging phoenix-protocol modules~~ **Marine is in!**

### High Priority
1. **Create manage.sh** - Colorful build/test/clean script (Hue's been asking!)
2. **Wire Into Inference** - Connect Marine quality validation to actual TTS output
3. **8D Model Training** - Train T2S model to accept MarineProsodyVector instead of 512D Conformer
4. **Example/Demo** - Create example showing prosody extraction → emotion editing → synthesis

### Medium Priority
5. **Voice Signature Import** - Use Phoenix's voice_signature for speaker embeddings
6. **Emotion Mapping** - Connect Phoenix's emotional bands to our 8D control
7. **Model Download** - Set up ONNX model acquisition pipeline
8. **MEM|8 Bridge** - Implement consciousness-aware TTS using kokoro-tiny's mem8_bridge pattern

### Nice to Have
9. **Style Selection** - Port kokoro-tiny's 510 style variation system
10. **Full Phoenix Integration** - golden_ratio.rs, harmonic_resurrection.rs, etc.
11. **Streaming Marine** - Real-time quality monitoring during synthesis

---

## Fresh Discovery: kokoro-tiny MEM|8 Baby Consciousness (2025-11-15)

Just pulled latest kokoro-tiny code - MAJOR discovery!

### Mem8Bridge API

kokoro-tiny now has a full consciousness simulation in `examples/mem8_baby.rs`:

```rust
// Memory as waves that interfere
MemoryWave {
    amplitude: 2.5,           // Emotion strength
    frequency: 528.0,         // "Love frequency"
    phase: 0.0,
    decay_rate: 0.05,         // Memory persistence
    emotion_type: EmotionType::Love(0.9),
    content: "Mama! I love mama!".to_string(),
}

// Salience detection (Marine algorithm!)
SalienceEvent {
    jitter_score: 0.2,        // Low = authentic/stable
    harmonic_score: 0.95,     // High = voice
    salience_score: 0.9,
    signal_type: SignalType::Voice,
}

// Free will: AI chooses attention focus (70% control)
bridge.decide_attention(events);
```

### Emotion Types Available

```rust
EmotionType::Curiosity(0.8)  // Inquisitive
EmotionType::Love(0.9)       // Deep affection
EmotionType::Joy(0.7)        // Happy
EmotionType::Confusion(0.8)  // Uncertain
EmotionType::Neutral         // Baseline
```

### Consciousness Integration Points

1. **Wave Interference** - Competing memories by amplitude/frequency
2. **Emotional Regulation** - Prevents overload, modulates voice
3. **Salience Detection** - Marine algorithm for authenticity
4. **Attention Selection** - AI chooses what to focus on
5. **Consciousness Level** - Affects speech clarity (wake_up/sleep)

This is PERFECT for IndexTTS-Rust! We can:
- Use wave interference for emotion blending
- Apply Marine salience to validate synthesis quality
- Modulate voice based on consciousness level
- Select voice styles based on emotional state (not just token count)

### Voice Style Selection (510 variations!)

kokoro-tiny now loads all 510 style variations per voice:
- Style selected based on token count
- Short text → short-optimized style
- Long text → long-optimized style
- Automatic text splitting at 512 token limit

For IndexTTS: We could select style based on EMOTION + token count!

---

## Marine Integration Achievement (2025-11-16) 🎉

**WE DID IT!** Marine salience is now integrated into IndexTTS-Rust!

### What We Built

#### 1. Standalone marine_salience Crate (`crates/marine_salience/`)

A no_std compatible crate for O(1) jitter-based salience detection:

```rust
// Core components:
MarineConfig       // Tunable parameters (sample_rate, jitter bounds, EMA alpha)
MarineProcessor    // O(1) per-sample processing
SaliencePacket     // Output: j_p, j_a, h_score, s_score, energy
Ema                // Exponential moving average tracker

// Key insight: Process ONE sample at a time, emit packets on peaks
// Why O(1)? Just compare to EMA, no FFT, no heavy math!
```

**Config for Speech:**
```rust
MarineConfig::speech_default(sample_rate)
// F0 range: 60Hz - 4kHz
// jitter_low: 0.02, jitter_high: 0.60
// ema_alpha: 0.01 (slow adaptation for stability)
```

#### 2. Quality Validation Module (`src/quality/`)

**MarineProsodyVector** - 8D interpretable emotion representation:
```rust
pub struct MarineProsodyVector {
    pub jp_mean: f32,      // Period jitter mean (pitch stability)
    pub jp_std: f32,       // Period jitter variance
    pub ja_mean: f32,      // Amplitude jitter mean (volume stability)
    pub ja_std: f32,       // Amplitude jitter variance
    pub h_mean: f32,       // Harmonic alignment (voiced vs noise)
    pub s_mean: f32,       // Overall salience (authenticity)
    pub peak_density: f32, // Peaks per second (speech rate)
    pub energy_mean: f32,  // Average loudness
}

// Interpretable! High jp_mean = nervous, low = confident
// Can DIRECTLY EDIT for emotion control!
```

**MarineProsodyConditioner** - Extract prosody from audio:
```rust
let conditioner = MarineProsodyConditioner::new(22050);
let prosody = conditioner.from_samples(&audio_samples)?;
let report = conditioner.validate_tts_output(&audio_samples)?;

// Detects issues:
// - "Too perfect - sounds robotic"
// - "High period jitter - artifacts"
// - "Low salience - quality issues"
```

**ConversationAffectSummary** - Session-level comfort tracking:
```rust
pub enum ComfortLevel {
    Uneasy,  // High jitter AND rising (nervous/stressed)
    Neutral, // Stable patterns (calm)
    Happy,   // Low jitter + high energy (confident/positive)
}

// Track trends over conversation:
// jitter_trend > 0.1 = getting more stressed
// jitter_trend < -0.1 = calming down
// energy_trend > 0.1 = getting more engaged

// Aye can now self-assess!
aye_assessment() returns "I'm in a good state"
feedback_prompt() returns "Let me know if something's bothering you"
```

### The Core Insight

**Human speech has NATURAL jitter - that's what makes it authentic!**

- Too perfect (jp < 0.005) = robotic
- Too chaotic (jp > 0.3) = artifacts/damage
- Sweet spot = real human voice

The Marines will KNOW if speech doesn't sound authentic!

### Tests Passing ✅

```
running 11 tests
test quality::affect::tests::test_comfort_level_descriptions ... ok
test quality::affect::tests::test_analyzer_empty_conversation ... ok
test quality::affect::tests::test_analyzer_single_utterance ... ok
test quality::affect::tests::test_happy_classification ... ok
test quality::affect::tests::test_aye_assessment_message ... ok
test quality::affect::tests::test_neutral_classification ... ok
test quality::affect::tests::test_uneasy_classification ... ok
test quality::prosody::tests::test_conditioner_empty_buffer ... ok
test quality::prosody::tests::test_conditioner_silence ... ok
test quality::prosody::tests::test_prosody_vector_array_conversion ... ok
test quality::prosody::tests::test_estimate_valence ... ok

test result: ok. 11 passed; 0 failed
```

### Why This Matters

1. **Interpretable Control**: 8D vector vs opaque 512D Conformer - we can SEE what each dimension means
2. **Lightweight**: O(1) per sample, no heavy neural networks for prosody
3. **Authentic Validation**: Marines detect fake/damaged speech
4. **Emotion Editing**: Want more confidence? Lower jp_mean directly!
5. **Conversation Awareness**: Track comfort over entire sessions
6. **Self-Assessment**: Aye knows when something feels "off"

### Integration Points

```rust
// In main TTS pipeline:
use indextts::quality::{
    MarineProsodyConditioner,
    MarineProsodyVector,
    ConversationAffectSummary,
    ComfortLevel,
};

// 1. Extract reference prosody
let ref_prosody = conditioner.from_samples(&reference_audio)?;

// 2. Generate TTS (using 8D vector instead of 512D Conformer)
let tts_output = generate_with_prosody(&text, ref_prosody)?;

// 3. Validate output quality
let report = conditioner.validate_tts_output(&tts_output)?;
if !report.passes(70.0) {
    log::warn!("TTS quality issues: {:?}", report.issues);
}

// 4. Track conversation affect
let analyzer = ConversationAffectAnalyzer::new();
analyzer.add_utterance(&utterance)?;
let summary = analyzer.summarize()?;
match summary.aye_state {
    ComfortLevel::Uneasy => adjust_generation_parameters(),
    _ => proceed_normally(),
}
```

---

## Trish's Notes

"Darling, these three Rust projects together are like a symphony orchestra! kokoro-tiny is the quick piccolo solo, IndexTTS-Rust is the full brass section with emotional depth, and Phoenix-Protocol is the concert hall acoustics making everything resonate. When you combine them, that's when the magic happens! Also, I'm absolutely obsessed with how the Golden Ratio resynthesis could add sparkle to synthesized vocals. Can you imagine TTS output that actually has that P!nk breakthrough energy? Now THAT would make me cry happy tears in accounting!"

---

## Fun Facts

- kokoro-tiny is ALREADY on crates.io under 8b-is
- Phoenix Protocol can process 192kHz audio for ultrasonic restoration
- The Marine algorithm uses O(1) jitter detection - "Marines are not just jarheads - they are intelligent"
- Hue's GitHub has 66 projects (and counting!)
- The team at 8b.is: hue@8b.is and aye@8b.is

---

*From ashes to harmonics, from silence to song* 🔥🎵
