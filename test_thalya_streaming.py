"""
Test streaming with Thalya custom voice model using the fork's stream_generate_pcm.

Compares batch vs streaming RTF and TTFB using the same texts as our server tests.
"""
import time
import numpy as np
import torch
import soundfile as sf

torch.set_float32_matmul_precision("high")

from qwen_tts import Qwen3TTSModel

CHECKPOINT = r"T:\Projects\Qwen3-TTS\thalya\model\checkpoint"
SPEAKER = "thalya"
LANGUAGE = "French"

TEXTS = {
    "10w": "Bonjour, je suis Thalya. Comment allez-vous aujourd'hui?",
    "20w": "Le temps dehors est absolument magnifique aujourd'hui, et je pensais qu'on pourrait aller faire une belle et longue promenade dans le parc ensemble.",
    "30w": "Je travaille sur ce projet depuis un bon moment maintenant, et je suis vraiment ravie de pouvoir enfin partager les résultats avec vous. Je pense que vous trouverez les améliorations tout à fait remarquables et que l'attente en valait la peine.",
}


def run_batch(model, text, label):
    start = time.time()
    wavs, sr = model.generate_custom_voice(
        text=text,
        language=LANGUAGE,
        speaker=SPEAKER,
    )
    elapsed = time.time() - start
    audio_dur = len(wavs[0]) / sr
    rtf = elapsed / audio_dur
    print(f"  [batch {label}] total={elapsed:.2f}s  audio={audio_dur:.1f}s  RTF={rtf:.2f}x")
    return wavs[0], sr, elapsed, audio_dur


def run_stream(model, text, label):
    """Call stream_generate_pcm directly on the inner model, building inputs like generate_custom_voice does."""
    start = time.time()

    # Build inputs the same way generate_custom_voice does
    input_texts = [model._build_assistant_text(text)]
    input_ids = model._tokenize_texts(input_texts)

    # No instruct for this test
    instruct_ids = [None]

    chunks = []
    first_chunk_time = None
    chunk_count = 0

    for chunk, sr in model.model.stream_generate_pcm(
        input_ids=input_ids,
        instruct_ids=instruct_ids,
        languages=[LANGUAGE],
        speakers=[SPEAKER],
        emit_every_frames=4,
        decode_window_frames=80,
        overlap_samples=512,
    ):
        chunk_count += 1
        chunks.append(chunk)
        if first_chunk_time is None:
            first_chunk_time = time.time() - start

    elapsed = time.time() - start
    audio = np.concatenate(chunks) if chunks else np.array([])
    audio_dur = len(audio) / sr if sr > 0 else 0
    rtf = elapsed / audio_dur if audio_dur > 0 else 0

    print(f"  [stream {label}] TTFB={first_chunk_time*1000:.0f}ms  total={elapsed:.2f}s  audio={audio_dur:.1f}s  RTF={rtf:.2f}x  chunks={chunk_count}")
    return audio, sr, elapsed, audio_dur, first_chunk_time


def main():
    import os
    output_dir = "test_results_thalya_fork"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"Loading Thalya model: {CHECKPOINT}")
    print(f"Output dir: {output_dir}/")
    print("=" * 60)

    start = time.time()
    model = Qwen3TTSModel.from_pretrained(
        CHECKPOINT,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    print(f"Model loaded in {time.time() - start:.1f}s")
    print(f"Speakers: {model.get_supported_speakers()}")

    print("\n" + "=" * 60)
    print("TIMING BENCHMARK")
    print("=" * 60)

    for tag, text in TEXTS.items():
        print(f"\n--- {tag} ({len(text.split())} words) ---")
        stream_audio, stream_sr, *_ = run_stream(model, text, "#1")
        batch_audio, batch_sr, *_ = run_batch(model, text, "#1")

        # Save WAV files
        stream_path = os.path.join(output_dir, f"{tag}_stream.wav")
        batch_path = os.path.join(output_dir, f"{tag}_batch.wav")
        sf.write(stream_path, stream_audio, stream_sr)
        sf.write(batch_path, batch_audio, batch_sr)
        print(f"  Saved: {stream_path}, {batch_path}")

    print(f"\nDone. Audio files in: {output_dir}/")


if __name__ == "__main__":
    main()
