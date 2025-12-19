# 🌍 Voice Translator App

A powerful multilingual voice and text translator built with LangGraph, OpenAI, and Google Text-to-Speech. Translate between multiple languages using either speech or text input, with audio output capabilities.

## ✨ Features

- **Dual Input Modes**: Choose between speech (microphone) or text input
- **5 Supported Languages**: English, Hindi, Tamil, Malayalam, and French
- **AI-Powered Translation**: Uses OpenAI's GPT-4 for accurate translations
- **Text-to-Speech Output**: Generates audio files of translated text
- **LangGraph Workflow**: Built with state-of-the-art workflow orchestration
- **Error Handling**: Robust error handling with fallback options

## 🎯 Use Cases

- Real-time language translation
- Language learning assistance
- Cross-cultural communication
- Voice note translation
- Multilingual content creation


## 🚀 Installation

### Running the Application

```bash
python voice_translator.py
```

### Step-by-Step Guide

1. **Select Input Mode**:
   - Option 1: Speech (speak into microphone)
   - Option 2: Text (type your text)

2. **Choose Input Language**:
   - 1. English
   - 2. Hindi
   - 3. Tamil
   - 4. Malayalam
   - 5. French

3. **Choose Output Language**:
   - Select from the same list of languages

4. **Provide Input**:
   - For **Speech Mode**: Speak clearly into your microphone when prompted
   - For **Text Mode**: Type your text when prompted

5. **Get Results**:
   - View the original and translated text
   - Audio file is automatically generated and played
   - Audio file is saved with timestamp (e.g., `translation_20241212_143025.mp3`)

## 📖 Example Usage

### Example 1: English to Hindi (Speech)

```
Select INPUT MODE:
  1. Speech (speak into microphone)
  2. Text (type your text)

Enter choice (1-2): 1

Select INPUT language:
  1. English
  [...]

Enter choice (1-5): 1

Select OUTPUT language:
  [...]
  2. Hindi

Enter choice (1-5): 2

🎤 Converting speech to text...
Listening in English... Speak now!
✓ Recognized: Hello, how are you?

🔄 Translating from English to Hindi...
✓ Translated: नमस्ते, आप कैसे हैं?

🔊 Converting text to speech in Hindi...
✓ Audio saved: translation_20241212_143025.mp3

✅ TRANSLATION COMPLETED!
```

### Example 2: Tamil to English (Text)

```
Select INPUT MODE:
  1. Speech (speak into microphone)
  2. Text (type your text)

Enter choice (1-2): 2

Select INPUT language:
  [...]
  3. Tamil

Enter choice (1-5): 3

Select OUTPUT language:
  1. English
  [...]

Enter choice (1-5): 1

⌨️  Text input mode activated...
Enter text in Tamil: வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?

✓ Received: வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?

🔄 Translating from Tamil to English...
✓ Translated: Hello, how are you?

🔊 Converting text to speech in English...
✓ Audio saved: translation_20241212_143130.mp3

✅ TRANSLATION COMPLETED!
```
### Workflow Components

1. **Input Routing**: Directs to speech or text input based on user selection
2. **Speech-to-Text**: Converts audio to text using Google Speech Recognition
3. **Text Input**: Direct text input from keyboard
4. **Translation**: Uses OpenAI GPT-4 for accurate language translation
5. **Text-to-Speech**: Converts translated text to audio using gTTS
6. **Error Handling**: Conditional edges handle errors at each step

## 🔧 Configuration

### Supported Languages

The app currently supports:

- **English** (en)
- **Hindi** (hi)
- **Tamil** (ta)
- **Malayalam** (ml)
- **French** (fr)


## 📁 Project Structure

```
voice-translator/
│
├── voice_translator.py    # Main application file
├── .env                   # Environment variables (API keys)
├── .gitignore            # Git ignore file
├── README.md             # This file
└── translation_*.mp3     # Generated audio files
```
