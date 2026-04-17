#!/usr/bin/env python3
"""
Turn a reference audio file into a "voice profile" directory that the
chatterbox binary can load with --ref-dir.

A voice profile is the full set of conditioning tensors the Chatterbox
pipeline consumes at inference time:

    <out>/speaker_emb.npy               (256,)          float32  T3 voice identity
    <out>/cond_prompt_speech_tokens.npy (375,)          int32    T3 prompt tokens
    <out>/embedding.npy                 (192,)          float32  S3Gen voice identity
    <out>/prompt_token.npy              (N,)            int32    S3Gen prompt tokens
    <out>/prompt_feat.npy               (mel_len, 80)   float32  S3Gen prompt mel

The first two override the T3 GGUF's built-in voice (chatterbox/builtin/*);
the last three override the S3Gen GGUF's built-in voice (s3gen/builtin/*).

Run it once per voice; reuse the directory:

    python scripts/prepare-voice.py --ref-audio me.wav --out voices/me/
    ./build/chatterbox --model models/chatterbox-t3-turbo.gguf \\
                       --s3gen-gguf models/chatterbox-s3gen.gguf \\
                       --ref-dir voices/me/ \\
                       --text "Hello in my voice." \\
                       --out me.wav

This is a Python-side bridge; VoiceEncoder, the mel extractor, and
S3TokenizerV2 will get C++ ports in follow-ups (A1 phase 2+).

Requires the chatterbox-ref virtualenv set up per README (torch, numpy,
safetensors, librosa, s3tokenizer).
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from chatterbox.tts_turbo import ChatterboxTurboTTS


def parse_args():
    p = argparse.ArgumentParser(
        description="Produce a voice profile from a reference audio file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--ref-audio", type=Path, required=True,
                   help="Reference audio (wav / flac / mp3). Any sample rate; "
                        "librosa resamples it. Needs > 5 s of clean speech.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output voice-profile directory.")
    p.add_argument("--device", default="cpu", help="Torch device (default: cpu)")
    p.add_argument("--exaggeration", type=float, default=0.5,
                   help="Emotion-advance knob (0..1). Default 0.5.")
    p.add_argument("--norm-loudness", action="store_true", default=True,
                   help="Normalise reference audio loudness before analysis.")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading Chatterbox Turbo on {args.device}...")
    tts = ChatterboxTurboTTS.from_pretrained(device=args.device)

    print(f"Analyzing {args.ref_audio}...")
    tts.prepare_conditionals(
        str(args.ref_audio),
        exaggeration=args.exaggeration,
        norm_loudness=args.norm_loudness,
    )

    # tts.conds is now a Conditionals(t3_cond, s3gen_ref_dict)
    t3_cond = tts.conds.t3
    s3_ref  = tts.conds.gen

    def to_np(t, dtype=np.float32):
        if isinstance(t, np.ndarray):
            a = t
        else:
            a = t.detach().cpu().numpy()
        a = np.ascontiguousarray(a)
        return a.astype(dtype, copy=False)

    # T3 side
    speaker_emb    = to_np(t3_cond.speaker_emb).reshape(-1)           # (256,)
    cond_prompt    = to_np(t3_cond.cond_prompt_speech_tokens, np.int32).reshape(-1)  # (375,)

    # S3Gen side
    embedding      = to_np(s3_ref["embedding"]).reshape(-1)           # (192,)
    prompt_token   = to_np(s3_ref["prompt_token"], np.int32).reshape(-1)
    prompt_feat    = to_np(s3_ref["prompt_feat"])
    if prompt_feat.ndim == 3:
        prompt_feat = prompt_feat[0]   # (mel_len, 80)

    np.save(args.out / "speaker_emb.npy",                speaker_emb)
    np.save(args.out / "cond_prompt_speech_tokens.npy",  cond_prompt)
    np.save(args.out / "embedding.npy",                  embedding)
    np.save(args.out / "prompt_token.npy",               prompt_token)
    np.save(args.out / "prompt_feat.npy",                prompt_feat)

    print(f"\nVoice profile written to {args.out}/")
    for name, arr in [
        ("speaker_emb.npy",                speaker_emb),
        ("cond_prompt_speech_tokens.npy",  cond_prompt),
        ("embedding.npy",                  embedding),
        ("prompt_token.npy",               prompt_token),
        ("prompt_feat.npy",                prompt_feat),
    ]:
        print(f"  {name:33s} {str(arr.shape):15s} {arr.dtype}")
    print()
    print("Use it with:")
    print(f"  ./build/chatterbox --model models/chatterbox-t3-turbo.gguf \\")
    print(f"                     --s3gen-gguf models/chatterbox-s3gen.gguf \\")
    print(f"                     --ref-dir {args.out} \\")
    print(f"                     --text 'Hello in my voice.' \\")
    print(f"                     --out out.wav")


if __name__ == "__main__":
    main()
