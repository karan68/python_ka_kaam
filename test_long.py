import whisper
from indic_transliteration import sanscript
from indic_transliteration.detect import detect
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
import os
from datetime import timedelta
import re
from transformers import WhisperTokenizer, WhisperForConditionalGeneration
from transformers import pipeline
from tqdm import tqdm
import time
from pathlib import Path
import concurrent.futures
from dataclasses import dataclass
from typing import Optional, Callable, List, Dict
import librosa
import numpy as np
import torch
os.environ["OMP_DISPLAY_ENV"] = "FALSE"

@dataclass
class ProcessingProgress:
    """Data class to track processing progress"""
    total_files: int = 0
    current_file: int = 0
    current_file_name: str = ""
    current_stage: str = ""
    progress: float = 0.0
    stage_progress: float = 0.0

class IndicCaptionGenerator:
    def __init__(self, huggingface_token, target_script='devanagari', 
                 include_romanization=True, confidence_threshold=0.7,
                 max_words_per_line=5,
                 min_silence_duration=0.3,
                 min_sustained_note_duration=1.0,
                 background_noise_threshold=0.05,
                 min_segment_duration=1.0,
                 progress_callback: Optional[Callable[[ProcessingProgress], None]] = None):
        """Initialize with dual track support"""
        # Previous initialization code remains the same
        self.progress = ProcessingProgress()
        self.progress_callback = progress_callback
        self.max_words_per_line = max_words_per_line
        self.min_silence_duration = min_silence_duration
        self.min_sustained_note_duration = min_sustained_note_duration
        self.background_noise_threshold = background_noise_threshold
        self.min_segment_duration = min_segment_duration
        
        self.setup_authentication(huggingface_token)
        self.init_config(target_script, include_romanization, confidence_threshold)
        self.init_script_mappings()
        self.init_models()

    def setup_authentication(self, token):
        """Set up Hugging Face authentication"""
        self.huggingface_token = token
        os.environ["HUGGINGFACE_TOKEN"] = token

    def init_config(self, target_script, include_romanization, confidence_threshold):
        """Initialize configuration parameters"""
        self.target_script = target_script
        self.include_romanization = include_romanization
        self.confidence_threshold = confidence_threshold

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

    def init_models(self):
        """Initialize the Whisper model and tokenizer"""
        self.update_progress(stage="Initializing models")
        
        # Initialize model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model("medium", device=device)
        
        # Initialize pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-medium",
            token=self.huggingface_token,
            chunk_length_s=30,
            device=device
        )

    def update_progress(self, stage=None, progress=None, 
                       stage_progress=None, file_name=None):
        """Update progress and notify callback"""
        if stage:
            self.progress.current_stage = stage
        if progress is not None:
            self.progress.progress = progress
        if stage_progress is not None:
            self.progress.stage_progress = stage_progress
        if file_name:
            self.progress.current_file_name = file_name

        if self.progress_callback:
            self.progress_callback(self.progress)

    def detect_silence_and_music(self, audio_path):
        """Enhanced detection of silent periods, musical segments, and sustained notes"""
        y, sr = librosa.load(audio_path)
        
        # Compute various audio features
        S = np.abs(librosa.stft(y))
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        pitch, magnitudes = librosa.piptrack(y=y, sr=sr)
        
        # Detect silence with dynamic thresholding
        non_silent = librosa.effects.split(
            y,
            top_db=30,  # Increased threshold for better silence detection
            frame_length=4096,  # Increased for better frequency resolution
            hop_length=1024
        )
        
        # Detect sustained notes
        sustained_notes = []
        pitches = np.mean(pitch, axis=0)
        pitch_stability = np.diff(pitches)
        stable_regions = np.where(np.abs(pitch_stability) < 0.1)[0]
        
        current_start = None
        for i in range(len(stable_regions) - 1):
            if current_start is None:
                current_start = stable_regions[i]
            elif stable_regions[i + 1] - stable_regions[i] > 1:
                duration = (stable_regions[i] - current_start) * 1024 / sr
                if duration >= self.min_sustained_note_duration:
                    sustained_notes.append({
                        'start': current_start * 1024 / sr,
                        'end': stable_regions[i] * 1024 / sr
                    })
                current_start = None
        
        # Enhanced silence detection
        silence_segments = []
        last_end = 0
        for start, end in non_silent:
            if start > last_end:
                # Check if this silence corresponds to a musical pause
                segment_energy = librosa.feature.rms(
                    y=y[last_end:start],
                    frame_length=2048,
                    hop_length=512
                )
                if np.mean(segment_energy) < self.background_noise_threshold:
                    silence_segments.append({
                        'start': last_end / sr,
                        'end': start / sr,
                        'type': 'silence'
                    })
            last_end = end
        
        # Detect background music/noise segments
        background_segments = []
        rms_energy = librosa.feature.rms(y=y)[0]
        background_frames = np.where(
            (rms_energy > self.background_noise_threshold) &
            (rms_energy < self.background_noise_threshold * 3)
        )[0]
        
        if len(background_frames) > 0:
            current_start = background_frames[0]
            for i in range(1, len(background_frames)):
                if background_frames[i] - background_frames[i-1] > sr * 0.5:  # Gap of 0.5s
                    if (background_frames[i-1] - current_start) * 512 / sr >= 1.0:  # Min 1s duration
                        background_segments.append({
                            'start': current_start * 512 / sr,
                            'end': background_frames[i-1] * 512 / sr,
                            'type': 'background'
                        })
                    current_start = background_frames[i]
        
        return silence_segments, background_segments, sustained_notes

    def transcribe_audio(self, audio_path: str, language_code: str) -> Dict:
        """Transcribe audio using Whisper"""
        self.update_progress(stage="Transcribing audio", stage_progress=0)
        
        try:
            # Transcribe using Whisper model directly
            result = self.model.transcribe(
                audio_path,
                language=language_code,
                word_timestamps=True
            )
            
            # Process the result to match our expected format
            processed_result = {
                'chunks': []
            }
            
            # Convert word-level timestamps to chunks
            current_chunk = {
                'text': '',
                'words': [],
                'start': None,
                'end': None
            }
            
            for segment in result['segments']:
                if len(current_chunk['words']) >= self.max_words_per_line:
                    if current_chunk['text'].strip():
                        processed_result['chunks'].append({
                            'text': current_chunk['text'].strip(),
                            'timestamp': [current_chunk['start'], current_chunk['end']]
                        })
                    current_chunk = {
                        'text': '',
                        'words': [],
                        'start': None,
                        'end': None
                    }
                
                current_chunk['text'] += ' ' + segment['text']
                current_chunk['words'].extend(segment.get('words', []))
                if current_chunk['start'] is None:
                    current_chunk['start'] = segment['start']
                current_chunk['end'] = segment['end']
            
            # Add the last chunk if it exists
            if current_chunk['text'].strip():
                processed_result['chunks'].append({
                    'text': current_chunk['text'].strip(),
                    'timestamp': [current_chunk['start'], current_chunk['end']]
                })
            
            self.update_progress(stage="Transcription complete", stage_progress=100)
            return processed_result
            
        except Exception as e:
            print(f"Transcription error: {str(e)}")
            raise

    def process_scripts(self, transcription):
        """Process and convert scripts with improved handling of repetitions"""
        processed_segments = []
        last_text = None
        repetition_count = 0
        min_segment_duration = 2.0  # Minimum duration for a segment
        
        for segment in transcription['chunks']:
            text = segment['text'].strip()
            start, end = segment['timestamp']
            duration = end - start
            
            # Skip very short segments that might be artifacts
            if duration < 0.1:
                continue
                
            script, confidence = self.detect_script_with_confidence(text)
            
            # Handle repetitions
            if text == last_text:
                repetition_count += 1
                # Only add repeated text if it's significantly separated in time
                last_segment = processed_segments[-1] if processed_segments else None
                if last_segment and (start - last_segment['end']) >= min_segment_duration:
                    if confidence >= self.confidence_threshold:
                        text = self.convert_script(text, script, 
                                                self.language_script_map.get(self.target_script))
                    processed_segments.append({
                        'start': start,
                        'end': end,
                        'text': text,
                        'script': script
                    })
            else:
                repetition_count = 0
                if confidence >= self.confidence_threshold:
                    text = self.convert_script(text, script, 
                                            self.language_script_map.get(self.target_script))
                processed_segments.append({
                    'start': start,
                    'end': end,
                    'text': text,
                    'script': script
                })
                
            last_text = text
        
        return processed_segments
    
    def detect_audio_layers(self, audio_path):
        """Enhanced detection of primary and background audio layers"""
        y, sr = librosa.load(audio_path)
        
        # Compute audio features
        S = np.abs(librosa.stft(y))
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        
        # Detect background music segments
        background_segments = []
        
        # Compute RMS energy
        rms_energy = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        
        # Compute spectral contrast for music detection
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(spectral_contrast, axis=0)
        
        # Identify music segments using spectral contrast
        is_music = contrast_mean > np.mean(contrast_mean) + 0.5 * np.std(contrast_mean)
        
        # Convert frame-wise detection to time segments
        music_frames = np.where(is_music)[0]
        if len(music_frames) > 0:
            current_start = music_frames[0]
            
            for i in range(1, len(music_frames)):
                if music_frames[i] - music_frames[i-1] > sr * 0.5:  # Gap of 0.5s
                    duration = (music_frames[i-1] - current_start) * 512 / sr
                    if duration >= 1.0:  # Min 1s duration
                        background_segments.append({
                            'start': current_start * 512 / sr,
                            'end': music_frames[i-1] * 512 / sr,
                            'type': 'music'
                        })
                    current_start = music_frames[i]
            
            # Add the last segment
            duration = (music_frames[-1] - current_start) * 512 / sr
            if duration >= 1.0:
                background_segments.append({
                    'start': current_start * 512 / sr,
                    'end': music_frames[-1] * 512 / sr,
                    'type': 'music'
                })
        
        # Detect silence segments
        silence_segments = []
        silence_threshold = 0.1 * np.mean(rms_energy)
        is_silence = rms_energy < silence_threshold
        
        silence_frames = np.where(is_silence)[0]
        if len(silence_frames) > 0:
            current_start = silence_frames[0]
            
            for i in range(1, len(silence_frames)):
                if silence_frames[i] - silence_frames[i-1] > sr * 0.5:
                    duration = (silence_frames[i-1] - current_start) * 512 / sr
                    if duration >= self.min_silence_duration:
                        silence_segments.append({
                            'start': current_start * 512 / sr,
                            'end': silence_frames[i-1] * 512 / sr,
                            'type': 'silence'
                        })
                    current_start = silence_frames[i]
        
        return background_segments, silence_segments   
    
    def detect_song_pattern(self, audio_path):
        """Detect repeating patterns in the song"""
        y, sr = librosa.load(audio_path)
        
        # Compute chromagram
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        
        # Compute self-similarity matrix
        sim_matrix = librosa.segment.recurrence_matrix(chroma, mode='affinity')
        
        # Find repeated sections
        repeated_sections = []
        threshold = 0.95  # Similarity threshold
        
        for i in range(sim_matrix.shape[0]):
            for j in range(i + sr, sim_matrix.shape[1]):  # Look for repetitions at least 1 second apart
                if sim_matrix[i, j] > threshold:
                    # Found a potential repetition
                    start_time = librosa.frames_to_time(i, sr=sr)
                    end_time = librosa.frames_to_time(j, sr=sr)
                    duration = end_time - start_time
                    
                    # Only consider reasonably long sections (e.g., > 2 seconds)
                    if duration > 2.0:
                        repeated_sections.append({
                            'start': start_time,
                            'end': end_time,
                            'duration': duration
                        })
        
        return repeated_sections

    def detect_script_with_confidence(self, text):
        """Detect script of text with confidence score"""
        try:
            detected = detect(text)
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

    def create_caption_file(self, segments, background_segments, output_path):
        """Create SRT file with dual track support"""
        def format_timestamp(seconds):
            td = timedelta(seconds=float(seconds))
            hours = td.seconds // 3600
            minutes = (td.seconds % 3600) // 60
            seconds = td.seconds % 60
            milliseconds = td.microseconds // 1000
            return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            counter = 1
            
            # Combine and sort all segments by start time
            all_segments = []
            
            # Add primary audio segments
            for segment in segments:
                all_segments.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'],
                    'script': segment['script'],
                    'type': 'primary'
                })
            
            # Add background music segments
            for segment in background_segments:
                if segment['type'] == 'music':
                    all_segments.append({
                        'start': segment['start'],
                        'end': segment['end'],
                        'text': '[music]',
                        'script': None,
                        'type': 'background'
                    })
                elif segment['type'] == 'silence':
                    all_segments.append({
                        'start': segment['start'],
                        'end': segment['end'],
                        'text': '[...]',
                        'script': None,
                        'type': 'silence'
                    })
            
            # Sort segments by start time
            all_segments.sort(key=lambda x: (x['start'], x['type'] != 'primary'))
            
            # Write segments to file
            for segment in all_segments:
                start_time = format_timestamp(segment['start'])
                end_time = format_timestamp(segment['end'])
                
                f.write(f"{counter}\n")
                f.write(f"{start_time} --> {end_time}\n")
                
                if segment['type'] == 'primary':
                    formatted_text = segment['text']
                    if segment['text'] != '♪' and self.include_romanization and segment['script']:
                        romanized = transliterate(
                            segment['text'],
                            segment['script'],
                            sanscript.IAST
                        )
                        formatted_text = f"{segment['text']}\n{romanized}"
                else:
                    formatted_text = segment['text']
                
                f.write(f"{formatted_text}\n\n")
                counter += 1

    def adjust_timestamps(self, audio_path: str, segments, silence_segments, background_segments, sustained_notes):
        """Adjust timestamps with improved handling of song structure"""
        adjusted_segments = []
        repeated_sections = self.detect_song_pattern(audio_path)
        
        # Minimum duration for a valid segment
        min_segment_duration = self.min_segment_duration
        
        for segment in segments:
            start_time = segment['start']
            end_time = segment['end']
            text = segment['text']
            
            # Skip segments that are too short
            if end_time - start_time < min_segment_duration:
                continue
                
            # Check if this segment is part of a repeated section
            is_repeated = any(
                rs['start'] <= start_time <= rs['end']
                for rs in repeated_sections
            )
            
            # Process silence and musical elements
            for silence in silence_segments:
                if (silence['start'] <= end_time and 
                    silence['end'] >= start_time and 
                    silence['end'] - silence['start'] >= self.min_silence_duration):
                    
                    if text.strip():
                        adjusted_segments.append({
                            'start': start_time,
                            'end': silence['start'],
                            'text': text,
                            'script': segment['script']
                        })
                    
                    # Add musical pause marker
                    adjusted_segments.append({
                        'start': silence['start'],
                        'end': silence['end'],
                        'text': '♪',
                        'script': None
                    })
                    
                    start_time = silence['end']
            
            # Add remaining text if it exists and has sufficient duration
            if start_time < end_time and text.strip():
                duration = end_time - start_time
                if duration >= min_segment_duration:
                    adjusted_segments.append({
                        'start': start_time,
                        'end': end_time,
                        'text': text,
                        'script': segment['script']
                    })
        
        # Merge very close segments with the same text
        merged_segments = []
        for segment in adjusted_segments:
            if not merged_segments:
                merged_segments.append(segment)
                continue
                
            last_segment = merged_segments[-1]
            if (segment['text'] == last_segment['text'] and 
                segment['start'] - last_segment['end'] < 0.3):  # 300ms threshold
                last_segment['end'] = segment['end']
            else:
                merged_segments.append(segment)
        
        return merged_segments

    def process_single_file(self, audio_path: str, output_path: str, 
                          language_code='hi') -> bool:
        """Process a single audio file with dual track support"""
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            self.update_progress(stage="Starting processing", 
                               file_name=os.path.basename(audio_path))
            
            # Detect audio layers
            self.update_progress(stage="Detecting audio layers", stage_progress=25)
            background_segments, silence_segments = self.detect_audio_layers(audio_path)
            
            # Transcribe audio
            self.update_progress(stage="Transcribing audio", stage_progress=50)
            transcription = self.transcribe_audio(audio_path, language_code)
            
            # Process scripts
            self.update_progress(stage="Processing scripts", stage_progress=75)
            processed_segments = self.process_scripts(transcription)
            
            # Create caption file with both primary and background audio
            self.update_progress(stage="Creating captions", stage_progress=90)
            self.create_caption_file(processed_segments, background_segments, output_path)
            
            self.update_progress(stage="Complete", stage_progress=100)
            return True
            
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            return False

def progress_handler(progress: ProcessingProgress):
        """Handle progress updates"""
        print(f"\rFile {progress.current_file}/{progress.total_files}: "
            f"{progress.current_file_name} - {progress.current_stage} "
            f"({progress.stage_progress:.1f}%) - "
            f"Overall: {progress.progress:.1f}%", end="")

# Example usage
if __name__ == "__main__":
    HUGGINGFACE_TOKEN = "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"
    
    generator = IndicCaptionGenerator(
        huggingface_token=HUGGINGFACE_TOKEN,
        target_script='devanagari',
        include_romanization=True,
        max_words_per_line=5,
        min_silence_duration=0.5,
        background_noise_threshold=0.1,
        min_segment_duration=1.0, 
        progress_callback=progress_handler
    )
    

    
    # Process a single file
    generator.process_single_file(
        audio_path="/Users/karany/Desktop/Desktop/folder/python_ka_kaam/backgroundmusic_and_song.mp3",
        output_path="/Users/karany/Desktop/Desktop/folder/python_ka_kaam/output_captions_backgroundmusic_and_song.srt",
        language_code='hi'
    )