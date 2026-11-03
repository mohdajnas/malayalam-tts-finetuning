# Malayalam TTS Finetuning Dataset Creator

This repository handles the full end-to-end preprocessing, transcription, and chunking pipeline designed to construct a clean Malayalam Text-To-Speech (TTS) dataset from long-form audio files. 

## 📋 Project Directory Structure
```
malayalam_dataset/
├── raw_audio/          # Original long-form MP3/WAV recordings
├── processed/
│   ├── wavs/           # Final chunked 24kHz Mono 16-bit PCM audio samples
│   └── temp/           # Intermediary configurations and resampled cache
└── metadata/
    └── transcription_draft.csv  # Final metadata mapping audio clips to transcripts
```

## 🛠️ Tech Stack & Key Pipelines
1. **Audio Standardization:** Audio inputs are programmatically standardized using `ffmpeg` to 24 kHz, Mono channel, and 16-bit PCM WAV formats (`pcm_s16le`), complying with high-quality TTS pipeline criteria.
2. **Whisper Transcription & VAD Segmentation:** We utilize `faster-whisper` (`large-v3`/`small`) optimized via TensorRT/cuDNN execution providers to implement Voice Activity Detection (VAD) chunking. Segments below 1.5 seconds are filtered out, and padding is added around detected utterances to ensure natural starts and stops without word clippings.
3. **NeMo Integration Investigations:** Investigated the loading of custom hybrid CTC/RNNT Conformer checkpoints (`ai4bharat/indicconformer_stt_ml_hybrid_ctc_rnnt_large`) by mapping vocabulary configurations to aggregate tokenizer configurations dynamically.

## 🚀 Getting Started
To set up and replicate the data preparation pipeline:
1. Upload raw Malayalam audio recording blocks under `malayalam_dataset/raw_audio/`.
2. Install the necessary Whisper and audio manipulation library dependencies:
   ```bash
   pip install faster-whisper soundfile numpy pandas scipy tqdm
   ```
3. Run the processing and segmentation pipeline in the notebook to produce isolated speech units mapped inside `transcription_draft.csv`.

## 📝 Metadata Mapping Format
Our output dataset maps files under the following format:
- `filename`: Standardized chunked filename matching TTS naming formats.
- `text`: Cleaned transcript segment synthesized by Whisper.
- `corrected_text`: Placeholders designed for manual post-transcript review and annotations.
- `speaker_id`: Identifier for the speaker.
- `duration`: Audio segment duration in seconds.
- `review_status`: Tracking flag (`needs_review` or `approved`).

---
*Created automatically using Google Colab with integrated fast-whisper and NeMo pipelines.*
