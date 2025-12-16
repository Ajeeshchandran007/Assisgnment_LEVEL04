"""
Voice Translator App using LangGraph
Supports: English, Hindi, Tamil, Malayalam, French
Input modes: Speech OR Text
"""

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
import speech_recognition as sr
from gtts import gTTS
import os
from datetime import datetime
import operator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define the state schema
class TranslatorState(TypedDict):
    input_mode: str  # "speech" or "text"
    input_language: str
    output_language: str
    audio_input: str
    recognized_text: str
    translated_text: str
    output_audio_file: str
    error: str
    status: str

# Language configurations
LANGUAGES = {
    "English": {"code": "en", "gtts_code": "en"},
    "Hindi": {"code": "hi", "gtts_code": "hi"},
    "Tamil": {"code": "ta", "gtts_code": "ta"},
    "Malayalam": {"code": "ml", "gtts_code": "ml"},
    "French": {"code": "fr", "gtts_code": "fr"}
}

# Dynamic translation prompt generator
def get_translation_prompt(from_lang: str, to_lang: str, text: str) -> str:
    """Generate translation prompt dynamically for any language pair"""
    return f"Translate the following {from_lang} text to {to_lang}: {text}"

# Node Functions
def text_input_node(state: TranslatorState) -> TranslatorState:
    """Accept text input directly from user"""
    print(f"⌨️  Text input mode activated...")
    
    try:
        text = input(f"\nEnter text in {state['input_language']}: ").strip()
        
        if not text:
            return {
                **state,
                "error": "No text entered. Please try again.",
                "status": "error"
            }
        
        print(f"✓ Received: {text}")
        
        return {
            **state,
            "recognized_text": text,
            "status": "text_received",
            "error": ""
        }
        
    except Exception as e:
        return {
            **state,
            "error": f"Error in text input: {str(e)}",
            "status": "error"
        }

def speech_to_text_node(state: TranslatorState) -> TranslatorState:
    """Convert speech input to text"""
    print(f"🎤 Converting speech to text...")
    
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            print(f"Listening in {state['input_language']}... Speak now!")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
            
        # Get language code
        lang_code = LANGUAGES[state['input_language']]['code']
        
        # Recognize speech
        text = recognizer.recognize_google(audio, language=lang_code)
        print(f"✓ Recognized: {text}")
        
        return {
            **state,
            "recognized_text": text,
            "status": "speech_recognized",
            "error": ""
        }
        
    except sr.WaitTimeoutError:
        return {
            **state,
            "error": "No speech detected. Please try again.",
            "status": "error"
        }
    except sr.UnknownValueError:
        return {
            **state,
            "error": "Could not understand the audio. Please speak clearly.",
            "status": "error"
        }
    except Exception as e:
        return {
            **state,
            "error": f"Error in speech recognition: {str(e)}",
            "status": "error"
        }

def translate_text_node(state: TranslatorState) -> TranslatorState:
    """Translate text from input language to output language using OpenAI"""
    print(f"🔄 Translating from {state['input_language']} to {state['output_language']}...")
    
    # If same language, skip translation
    if state['input_language'] == state['output_language']:
        return {
            **state,
            "translated_text": state['recognized_text'],
            "status": "translated"
        }
    
    try:
        # Use OpenAI API for translation
        from openai import OpenAI
        
        # Read API key from environment variable (loaded from .env)
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            print("⚠️  OPENAI_API_KEY not found in .env file. Using fallback translation.")
            translated = f"[Translated to {state['output_language']}]: {state['recognized_text']}"
        else:
            client = OpenAI(api_key=api_key)
            
            prompt = get_translation_prompt(
                state['input_language'],
                state['output_language'],
                state['recognized_text']
            )
            
            # Call OpenAI API for translation
            response = client.chat.completions.create(
                model="gpt-4o",  # or "gpt-3.5-turbo" for faster/cheaper
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional translator. Provide ONLY the translated text without any explanation or additional text."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            translated = response.choices[0].message.content.strip()
        
        print(f"✓ Translated: {translated}")
        
        return {
            **state,
            "translated_text": translated,
            "status": "translated",
            "error": ""
        }
        
    except Exception as e:
        # Fallback to mock translation on error
        print(f"⚠️  Translation error: {e}. Using fallback.")
        translated = f"[Translated to {state['output_language']}]: {state['recognized_text']}"
        return {
            **state,
            "translated_text": translated,
            "status": "translated",
            "error": ""
        }

def text_to_speech_node(state: TranslatorState) -> TranslatorState:
    """Convert translated text to speech"""
    print(f"🔊 Converting text to speech in {state['output_language']}...")
    
    try:
        lang_code = LANGUAGES[state['output_language']]['gtts_code']
        
        # Create TTS object
        tts = gTTS(text=state['translated_text'], lang=lang_code, slow=False)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"translation_{timestamp}.mp3"
        
        # Save audio file
        tts.save(filename)
        print(f"✓ Audio saved: {filename}")
        
        # Play audio (cross-platform)
        play_audio(filename)
        
        return {
            **state,
            "output_audio_file": filename,
            "status": "completed",
            "error": ""
        }
        
    except Exception as e:
        return {
            **state,
            "error": f"Text-to-speech error: {str(e)}",
            "status": "error"
        }

def play_audio(filename: str):
    """Play audio file (cross-platform)"""
    try:
        if os.name == 'nt':  # Windows
            os.system(f'start {filename}')
        elif os.name == 'posix':  # macOS/Linux
            if os.uname().sysname == 'Darwin':  # macOS
                os.system(f'afplay {filename}')
            else:  # Linux
                os.system(f'mpg123 {filename}')
    except Exception as e:
        print(f"Could not play audio automatically: {e}")
        print(f"Audio file saved as: {filename}")

def should_continue_after_input(state: TranslatorState) -> str:
    """Routing function after text/speech input"""
    if state.get('error') or not state.get('recognized_text'):
        return "end"
    return "continue"

def should_continue_after_translate(state: TranslatorState) -> str:
    """Routing function after translation"""
    if state.get('error') or not state.get('translated_text'):
        return "end"
    return "continue"

def route_input_mode(state: TranslatorState) -> str:
    """Route to appropriate input method based on mode"""
    if state['input_mode'] == 'text':
        return "text_input"
    else:
        return "speech_input"

# Build the LangGraph workflow
def create_translator_graph():
    """Create the translation workflow graph"""
    
    workflow = StateGraph(TranslatorState)
    
    # Add nodes
    workflow.add_node("text_input", text_input_node)
    workflow.add_node("speech_input", speech_to_text_node)
    workflow.add_node("translate", translate_text_node)
    workflow.add_node("text_to_speech", text_to_speech_node)
    
    # Set entry point with conditional routing
    workflow.set_conditional_entry_point(
        route_input_mode,
        {
            "text_input": "text_input",
            "speech_input": "speech_input"
        }
    )
    
    # Add conditional edges with error handling
    workflow.add_conditional_edges(
        "text_input",
        should_continue_after_input,
        {
            "continue": "translate",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "speech_input",
        should_continue_after_input,
        {
            "continue": "translate",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "translate",
        should_continue_after_translate,
        {
            "continue": "text_to_speech",
            "end": END
        }
    )
    
    workflow.add_edge("text_to_speech", END)
    
    return workflow.compile()

def main():
    """Main function to run the voice translator"""
    print("=" * 60)
    print("🌍 VOICE TRANSLATOR APP (LangGraph)")
    print("=" * 60)
    print("\nSupported Languages:")
    for idx, lang in enumerate(LANGUAGES.keys(), 1):
        print(f"  {idx}. {lang}")
    print()
    
    # Get input mode
    print("Select INPUT MODE:")
    print("  1. Speech (speak into microphone)")
    print("  2. Text (type your text)")
    
    mode_choice = int(input("\nEnter choice (1-2): "))
    input_mode = "speech" if mode_choice == 1 else "text"
    
    # Get user input for languages
    print("\nSelect INPUT language:")
    for idx, lang in enumerate(LANGUAGES.keys(), 1):
        print(f"  {idx}. {lang}")
    
    input_choice = int(input("\nEnter choice (1-5): "))
    input_language = list(LANGUAGES.keys())[input_choice - 1]
    
    print("\nSelect OUTPUT language:")
    for idx, lang in enumerate(LANGUAGES.keys(), 1):
        print(f"  {idx}. {lang}")
    
    output_choice = int(input("\nEnter choice (1-5): "))
    output_language = list(LANGUAGES.keys())[output_choice - 1]
    
    print(f"\n✓ Input Mode: {input_mode.upper()}")
    print(f"✓ Input Language: {input_language}")
    print(f"✓ Output Language: {output_language}")
    print("\n" + "=" * 60)
    
    # Initialize state
    initial_state = {
        "input_mode": input_mode,
        "input_language": input_language,
        "output_language": output_language,
        "audio_input": "",
        "recognized_text": "",
        "translated_text": "",
        "output_audio_file": "",
        "error": "",
        "status": "initialized"
    }
    
    # Create and run the graph
    graph = create_translator_graph()
    
    try:
        result = graph.invoke(initial_state)
        
        if result.get('error'):
            print(f"\n❌ Error: {result['error']}")
        else:
            print("\n" + "=" * 60)
            print("✅ TRANSLATION COMPLETED!")
            print("=" * 60)
            print(f"\n📝 Original Text ({input_language}):")
            print(f"   {result['recognized_text']}")
            print(f"\n🌐 Translated Text ({output_language}):")
            print(f"   {result['translated_text']}")
            print(f"\n🔊 Audio File: {result['output_audio_file']}")
            print("\n" + "=" * 60)
            
    except Exception as e:
        print(f"\n❌ Application Error: {str(e)}")

if __name__ == "__main__":
    # Required installations:
    print("📦 Required packages:")
    print("   pip install langgraph speechrecognition pyaudio gtts openai python-dotenv")
    print()
    
    # Check if packages are installed
    try:
        import langgraph
        import speech_recognition
        from gtts import gTTS
        from openai import OpenAI
        from dotenv import load_dotenv
        print("✓ All required packages are installed!\n")
        
        # Check for OpenAI API key
        if not os.getenv('OPENAI_API_KEY'):
            print("⚠️  WARNING: OPENAI_API_KEY not found in .env file!")
            print("   Create a .env file in the same directory with:")
            print("   OPENAI_API_KEY=your-api-key-here")
            print("   The app will use fallback translation without API key.\n")
        else:
            print("✓ OpenAI API key loaded from .env file!\n")
        
        main()
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please install required packages using:")
        print("pip install langgraph speechrecognition pyaudio gtts openai python-dotenv")