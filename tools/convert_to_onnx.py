#!/usr/bin/env python3
"""
Convert IndexTTS-2 PyTorch models to ONNX format for Rust inference!

This script converts the three main models:
1. GPT model (gpt.pth) - Autoregressive text-to-semantic generation
2. S2Mel model (s2mel.pth) - Semantic-to-mel spectrogram conversion
3. BigVGAN - Mel-to-waveform vocoder (already available as ONNX from NVIDIA)

Usage:
    python tools/convert_to_onnx.py

Output:
    models/gpt.onnx
    models/s2mel.onnx
    models/bigvgan.onnx (if needed, otherwise use NVIDIA's)

Why ONNX?
    - Cross-platform: Works on Windows, Linux, macOS, M1/M2 Macs
    - Fast: ONNX Runtime is highly optimized
    - Rust-native: ort crate provides excellent ONNX Runtime bindings
    - No Python: Production inference without Python dependency hell!

Author: Aye & Hue @ 8b.is
"""

import os
import sys

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

# Set HF cache
os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'

print("=" * 70)
print("  IndexTTS-2 PyTorch to ONNX Converter")
print("  For Rust inference with ort crate!")
print("=" * 70)
print()

# Check for models
if not os.path.exists("checkpoints/gpt.pth"):
    print("ERROR: Models not found!")
    print("Run: python tools/download_files.py -s huggingface")
    sys.exit(1)

import torch
import torch.onnx
import numpy as np
from pathlib import Path

# Add reference code to path
sys.path.insert(0, "indextts - REMOVING - REF ONLY")

# Create output directory
output_dir = Path("models")
output_dir.mkdir(exist_ok=True)

print(f"PyTorch version: {torch.__version__}")
print(f"Output directory: {output_dir}")
print()


def export_speaker_encoder():
    """
    Export the CAM++ speaker encoder to ONNX.

    This model extracts speaker embeddings from reference audio.
    Input: mel spectrogram [batch, n_mels, time]
    Output: speaker embedding [batch, 192]
    """
    print("\n" + "=" * 50)
    print("Exporting Speaker Encoder (CAM++)")
    print("=" * 50)

    try:
        from omegaconf import OmegaConf
        from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus

        # Load config
        cfg = OmegaConf.load("checkpoints/config.yaml")

        # Create model
        model = CAMPPlus(feat_dim=80, embedding_size=192)

        # Load weights
        weights_path = "./checkpoints/hf_cache/models--funasr--campplus/snapshots/fb71fe990cbf6031ae6987a2d76fe64f94377b7e/campplus_cn_common.bin"
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(state_dict)
            print(f"Loaded weights from: {weights_path}")

        model.eval()

        # CAMPPlus expects [batch, time, n_mels] NOT [batch, n_mels, time]!
        # This is the key insight - the model processes time-series of mel features
        dummy_input = torch.randn(1, 100, 80)  # [batch, time, features]

        # Verify forward pass works before export
        with torch.no_grad():
            test_output = model(dummy_input)
            print(f"Forward pass works! Output shape: {test_output.shape}")

        # Export to ONNX
        output_path = output_dir / "speaker_encoder.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=['mel_spectrogram'],
            output_names=['speaker_embedding'],
            dynamic_axes={
                'mel_spectrogram': {0: 'batch', 1: 'time'},  # time is dim 1!
                'speaker_embedding': {0: 'batch'}
            },
            opset_version=18,  # Use 18+ for latest features
            do_constant_folding=True,
        )

        # Verify the export
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)

        print(f"✓ Exported: {output_path}")
        print(f"  Input: mel_spectrogram [batch, time, 80]")  # Corrected!
        print(f"  Output: speaker_embedding [batch, 192]")
        print(f"✓ ONNX model verified!")
        return True

    except Exception as e:
        print(f"✗ Failed to export speaker encoder: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_gpt_model():
    """
    Export the GPT autoregressive model to ONNX.

    This is the most complex model - generates semantic tokens from text.
    We may need to export it in parts due to KV caching.

    Input: text_tokens [batch, seq_len], speaker_embedding [batch, 192]
    Output: semantic_codes [batch, code_len]
    """
    print("\n" + "=" * 50)
    print("Exporting GPT Model (Autoregressive)")
    print("=" * 50)

    try:
        from omegaconf import OmegaConf

        # Load the full model config
        cfg = OmegaConf.load("checkpoints/config.yaml")

        # This is tricky - GPT models with KV caching are hard to export
        # We might need to:
        # 1. Export just the forward pass without caching
        # 2. Or export separate encoder/decoder parts

        print("GPT model export is complex due to:")
        print("  - Autoregressive generation with KV caching")
        print("  - Dynamic sequence lengths")
        print("  - Multiple internal components")
        print()
        print("Options:")
        print("  A) Export without KV cache (slower but simpler)")
        print("  B) Export encoder + single-step decoder (efficient)")
        print("  C) Use torch.compile + ONNX tracing")
        print()

        # For now, let's try the simpler approach
        from infer_v2 import IndexTTS2

        # Load model
        tts = IndexTTS2(
            cfg_path="checkpoints/config.yaml",
            model_dir="checkpoints",
            use_fp16=False,
            device="cpu"
        )

        # Get the GPT component
        gpt = tts.gpt
        gpt.eval()

        print(f"GPT model loaded: {type(gpt)}")
        print(f"Parameters: {sum(p.numel() for p in gpt.parameters()):,}")

        # The GPT model architecture:
        # - Text encoder (embeddings + transformer)
        # - Speaker conditioning
        # - Autoregressive decoder

        # Let's export the text encoder first
        output_path = output_dir / "gpt_encoder.onnx"

        # Create dummy inputs
        text_tokens = torch.randint(0, 30000, (1, 32), dtype=torch.int64)

        # This will likely fail due to complex control flow
        # but let's try!
        print(f"Attempting GPT export (may require modifications)...")

        # For now, just report what we learned
        print()
        print("Note: Full GPT export requires modifying the model code")
        print("to remove dynamic control flow. Creating a wrapper...")

        return False

    except Exception as e:
        print(f"✗ Failed to export GPT: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_s2mel_model():
    """
    Export the Semantic-to-Mel model (flow matching).

    This converts semantic codes to mel spectrograms.
    Input: semantic_codes [batch, code_len], speaker_embedding [batch, 192]
    Output: mel_spectrogram [batch, 80, mel_len]
    """
    print("\n" + "=" * 50)
    print("Exporting S2Mel Model (Flow Matching)")
    print("=" * 50)

    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load("checkpoints/config.yaml")

        print("S2Mel model (Diffusion/Flow Matching) is also complex:")
        print("  - Multiple denoising steps (iterative)")
        print("  - CFM (Conditional Flow Matching) requires ODE solving")
        print()
        print("Export strategy:")
        print("  1. Export the single denoising step")
        print("  2. Run iteration loop in Rust")
        print()

        return False

    except Exception as e:
        print(f"✗ Failed to export S2Mel: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_bigvgan():
    """
    Export BigVGAN vocoder to ONNX.

    Good news: NVIDIA provides pre-trained BigVGAN models!
    Even better: They're designed for easy ONNX export.

    Input: mel_spectrogram [batch, 80, mel_len]
    Output: waveform [batch, 1, wave_len]
    """
    print("\n" + "=" * 50)
    print("Exporting BigVGAN Vocoder")
    print("=" * 50)

    try:
        # BigVGAN from NVIDIA is easier to export
        # Let's check if we already have it

        print("BigVGAN options:")
        print("  1. Use NVIDIA's pre-exported ONNX (recommended)")
        print("     https://github.com/NVIDIA/BigVGAN")
        print()
        print("  2. Export from PyTorch weights (we'll do this)")
        print()

        # Try to load BigVGAN
        try:
            from bigvgan import bigvgan
            model = bigvgan.BigVGAN.from_pretrained(
                'nvidia/bigvgan_v2_22khz_80band_256x',
                use_cuda_kernel=False
            )
            model.eval()
            model.remove_weight_norm()  # Important for ONNX!

            print(f"BigVGAN loaded from HuggingFace")
            print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

            # Create dummy input
            dummy_mel = torch.randn(1, 80, 100)

            # Export
            output_path = output_dir / "bigvgan.onnx"
            torch.onnx.export(
                model,
                dummy_mel,
                str(output_path),
                input_names=['mel_spectrogram'],
                output_names=['waveform'],
                dynamic_axes={
                    'mel_spectrogram': {0: 'batch', 2: 'mel_length'},
                    'waveform': {0: 'batch', 2: 'wave_length'}
                },
                opset_version=18,  # Use 18+ for latest features
                do_constant_folding=True,
            )

            print(f"✓ Exported: {output_path}")
            print(f"  Input: mel_spectrogram [batch, 80, mel_len]")
            print(f"  Output: waveform [batch, 1, wave_len]")

            # Verify the export
            import onnx
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            print(f"✓ ONNX model verified!")

            return True

        except ImportError:
            print("bigvgan package not installed, installing...")
            os.system("pip install bigvgan")
            print("Please re-run the script.")
            return False

    except Exception as e:
        print(f"✗ Failed to export BigVGAN: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\nStarting ONNX conversion...\n")

    results = {}

    # Export each component
    results['speaker_encoder'] = export_speaker_encoder()
    results['gpt'] = export_gpt_model()
    results['s2mel'] = export_s2mel_model()
    results['bigvgan'] = export_bigvgan()

    # Summary
    print("\n" + "=" * 70)
    print("  CONVERSION SUMMARY")
    print("=" * 70)

    for name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ NEEDS WORK"
        print(f"  {name:20} {status}")

    print()

    if all(results.values()):
        print("All models converted! Ready for Rust inference.")
    else:
        print("Some models need manual intervention.")
        print()
        print("For complex models (GPT, S2Mel), consider:")
        print("  1. Modifying the Python code to remove dynamic control flow")
        print("  2. Using torch.jit.trace with concrete inputs")
        print("  3. Exporting subcomponents separately")
        print("  4. Using ONNX Runtime's transformer optimizations")

    print()
    print("Output directory:", output_dir.absolute())


if __name__ == "__main__":
    main()
