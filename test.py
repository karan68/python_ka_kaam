import whisper
from indic_transliteration import sanscript
from indic_transliteration.detect import detect
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
import os
from datetime import timedelta
import re
from transformers import WhisperTokenizer, WhisperForConditionalGeneration
from transformers import pipeline

class IndicCaptionGenerator:
    def __init__(self, huggingface_token, target_script='devanagari', include_romanization=True, confidence_threshold=0.7):
        """
        Initialize the caption generator with authentication
        
        Parameters:
            huggingface_token (str): Your Hugging Face API token
            target_script (str): Target script for output
            include_romanization (bool): Whether to include romanized text
            confidence_threshold (float): Threshold for script detection
        """
        # Set up authentication
        self.huggingface_token = huggingface_token
        os.environ["HUGGINGFACE_TOKEN"] = huggingface_token
        
        print("Initializing models...")
        # Initialize Whisper model from Hugging Face
        self.tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-medium", token=huggingface_token)
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium", token=huggingface_token)
        
        # Create ASR pipeline with correct parameters
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-medium",
            token=huggingface_token,
            chunk_length_s=30,
            return_timestamps="word"  # Changed to word-level timestamps
        )
        
        # Store configuration
        self.target_script = target_script
        self.include_romanization = include_romanization
        self.confidence_threshold = confidence_threshold
        
        # Initialize script mappings
        self.init_script_mappings()
        
    def init_script_mappings(self):
        """Initialize script and language mappings"""
        self.language_script_map = {
            'hi': sanscript.DEVANAGARI,
            'bn': sanscript.BENGALI,
            'gu': sanscript.GUJARATI,
            'pa': sanscript.GURMUKHI,
            'kn': sanscript.KANNADA,
            'ml': sanscript.MALAYALAM,
            'or': sanscript.ORIYA,
            'ta': sanscript.TAMIL,
            'te': sanscript.TELUGU
        }
        
        self.music_patterns = {
            'chorus': r'\b(चोरस|കോറസ്|கோரஸ்|కోరస్)\b',
            'verse': r'\b(अंतरा|വേഴ്സ്|வேர்ஸ்|వర్స్)\b',
            'interlude': r'\b(अंतरा|ഇന്റര്‍ല്യൂഡ്|இடைவேளை|ఇంటర్లూడ్)\b'
        }

    def detect_script_with_confidence(self, text):
        """Detect script of text with confidence score"""
        try:
            detected = detect(text)
            # Simple confidence calculation based on character ratio
            total_chars = len(text.strip())
            script_chars = sum(1 for char in text if char.isalpha())
            confidence = script_chars / total_chars if total_chars > 0 else 0
            return detected, confidence
        except Exception as e:
            print(f"Script detection error: {str(e)}")
            return None, 0.0

    def convert_script(self, text, from_script, to_script):
        """Convert text from one script to another"""
        try:
            if from_script and to_script and from_script != to_script:
                return transliterate(text, from_script, to_script)
            return text
        except Exception as e:
            print(f"Script conversion error: {str(e)}")
            return text

    def format_timestamp(self, seconds):
        """Convert seconds to SRT timestamp format"""
        td = timedelta(seconds=float(seconds))
        hours = td.seconds // 3600
        minutes = (td.seconds % 3600) // 60
        seconds = td.seconds % 60
        milliseconds = td.microseconds // 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def format_lyrics(self, text, detected_script):
        """Format lyrics with optional romanization"""
        if not self.include_romanization or not text.strip():
            return text
        
        try:
            romanized = transliterate(text, detected_script, sanscript.IAST)
            return f"{text}\n{romanized}"
        except Exception as e:
            print(f"Romanization error: {str(e)}")
            return text

    def transcribe_audio(self, audio_path, language_code):
        """Step 1: Transcribe audio using Whisper"""
        try:
            # Generate transcription with word-level timestamps
            result = self.pipe(
                audio_path,
                generate_kwargs={"language": language_code}  # Specify language in generate_kwargs
            )
            
            # Convert word-level timestamps to segment-level timestamps
            segments = []
            current_segment = {"text": "", "timestamp": [None, None]}
            
            for word_info in result["chunks"]:
                if not current_segment["timestamp"][0]:
                    current_segment["timestamp"][0] = word_info["timestamp"][0]
                
                current_segment["text"] += " " + word_info["text"]
                current_segment["timestamp"][1] = word_info["timestamp"][1]
                
                # Start new segment if text is long enough or pause is detected
                if len(current_segment["text"].split()) >= 5:
                    segments.append(current_segment)
                    current_segment = {"text": "", "timestamp": [None, None]}
            
            # Add last segment if not empty
            if current_segment["text"].strip():
                segments.append(current_segment)
            
            return {"chunks": segments}
            
        except Exception as e:
            print(f"Transcription error: {str(e)}")
            raise

    def process_scripts(self, transcription):
        """Step 2: Process and convert scripts for each segment"""
        processed_segments = []
        
        for segment in transcription['chunks']:
            # Detect script and convert if needed
            text = segment['text'].strip()
            script, confidence = self.detect_script_with_confidence(text)
            
            if confidence >= self.confidence_threshold:
                text = self.convert_script(
                    text,
                    script,
                    self.language_script_map.get(self.target_script)
                )
            
            processed_segments.append({
                'start': segment['timestamp'][0],
                'end': segment['timestamp'][1],
                'text': text,
                'script': script
            })
            
        return processed_segments

    def process_audio(self, audio_path, output_path, language_code='hi'):
        """
        Main processing flow for audio captioning
        """
        try:
            # 1. Transcription
            print("Step 1: Transcribing audio...")
            transcription = self.transcribe_audio(audio_path, language_code)
            
            # 2. Script Processing
            print("Step 2: Processing scripts...")
            processed_segments = self.process_scripts(transcription)
            
            # 3. Caption Formatting
            print("Step 3: Formatting captions...")
            self.create_caption_file(processed_segments, output_path)
            
            print(f"Caption file created successfully at: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error in processing flow: {str(e)}")
            return False

    def create_caption_file(self, segments, output_path):
        """Step 3: Create the final SRT caption file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, 1):
                # Format timestamps
                start_time = self.format_timestamp(segment['start'])
                end_time = self.format_timestamp(segment['end'])
                
                # Create caption entry
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                
                # Format lyrics with romanization if enabled
                formatted_lyrics = self.format_lyrics(
                    segment['text'],
                    segment['script']
                )
                f.write(f"{formatted_lyrics}\n\n")

# Example usage showing the complete flow
if __name__ == "__main__":
    # Your Hugging Face token
    HUGGINGFACE_TOKEN = "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"
    
    # Initialize the generator
    generator = IndicCaptionGenerator(
        huggingface_token=HUGGINGFACE_TOKEN,
        target_script='devanagari',
        include_romanization=True
    )
    
    # Process an audio file
    generator.process_audio(
        audio_path="/Users/karany/Desktop/Desktop/folder/python_ka_kaam/audio.mp3",
        output_path="output_captions_s.srt",
        language_code='hi'
    )