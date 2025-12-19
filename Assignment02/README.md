# Audio Similarity Finder

A Python-based audio similarity search system that uses advanced feature extraction and vector embeddings to find similar audio files. Optimized for voice/speaker recognition and audio matching using ChromaDB for efficient similarity search.

## Features

- **Comprehensive Audio Feature Extraction**: Extracts multiple audio features including MFCC, Mel Spectrogram, Spectral features, Chroma, Tonnetz, and rhythm features
- **Persistent Vector Database**: Uses ChromaDB for efficient storage and retrieval
- **Cosine Similarity Matching**: Fast and accurate similarity scoring
- **Batch Processing**: Add entire directories of audio files at once
- **Multiple Audio Format Support**: MP3, WAV, FLAC, OGG, M4A, AAC

## Requirements

### Dependencies

```bash
pip install librosa numpy chromadb scipy
```


Create necessary directories:
```bash
mkdir input query audio_database
```

## Project Structure

```
audio-similarity-finder/
│
├── AudioSimilarityFinder.py    # Main script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── input/                       # Directory for audio library files
│   ├── audio1.mp3
│   ├── audio2.wav
│   └── ...
│
├── query/                       # Directory for query audio files
│   └── query_audio.mp3
│
└── audio_database/              # ChromaDB storage (auto-created)
    └── chroma.sqlite3
```

## How to Test

### Basic Testing (Quick Start)

1. **Prepare your audio files**:
   - Place audio files you want to search through in the `./input` directory
   - Place the audio file you want to find matches for in the `./query` directory( For testing, please ensure that you need to rename audio file to query_audio.mp3)

2. **Run the script**:
```bash
python AudioSimilarityFinder.py
```

3. **View results**: The script will:
   - Add all audio files from `./input` to the database
   - Search for files similar to `./query/query_audio.mp3`
   - Display similarity scores and the best matches


The system provides similarity scores from 0-100%:

- **80-100%**: Very High Similarity - Likely same speaker/source
- **60-79%**: High Similarity - Very similar characteristics
- **40-59%**: Moderate Similarity - Some common features
- **0-39%**: Low Similarity - Different characteristics

### Example Output

```
======================================================================
Query Audio: my_voice.mp3
======================================================================

🎯 Best Match:
----------------------------------------------------------------------
✓ reference_voice_1.mp3
   Similarity Score: 87.45%
   Cosine Distance: 0.125500
   Path: ./input/reference_voice_1.mp3
   Interpretation: Very High Similarity - Likely same speaker/source
```

