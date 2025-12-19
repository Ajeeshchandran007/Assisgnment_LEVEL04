import librosa
import numpy as np
import chromadb
from chromadb.config import Settings
import os
from pathlib import Path
from typing import List, Tuple, Dict
import warnings
from scipy.stats import skew, kurtosis

warnings.filterwarnings('ignore')


class AudioSimilaritySearch:
    """
    Enhanced audio similarity search system with improved feature extraction
    for better voice/speaker recognition and audio matching.
    """
    
    def __init__(self, collection_name: str = "audio_collection", persist_directory: str = "./chroma_db"):
        """
        Initialize the Audio Similarity Search system.
        
        Args:
            collection_name: Name for the ChromaDB collection
            persist_directory: Directory to persist the database
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "Audio file embeddings for similarity search",
                "hnsw:space": "cosine"
            }
        )
        
        print(f"Initialized AudioSimilaritySearch with collection: {collection_name}")
        print(f"Using cosine similarity metric")
        print(f"Current database size: {self.collection.count()} audio files")
    
    def extract_audio_features(self, audio_path: str, n_mfcc: int = 40) -> np.ndarray:
        """
        Extract comprehensive audio features optimized for voice/speaker recognition.
        
        Args:
            audio_path: Path to the audio file
            n_mfcc: Number of MFCC coefficients to extract
            
        Returns:
            Normalized combined feature vector representing the audio
        """
        try:
            y, sr = librosa.load(audio_path, sr=22050, duration=60, mono=True)
            
            if len(y) == 0:
                print(f"No audio data in {audio_path}")
                return None
            
            y = librosa.util.normalize(y)
            y, _ = librosa.effects.trim(y, top_db=20)
            
            if len(y) == 0:
                print(f"No audio after trimming silence: {audio_path}")
                return None
            
            # MFCC Features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            mfcc_delta = np.mean(librosa.feature.delta(mfcc), axis=1)
            mfcc_delta2 = np.mean(librosa.feature.delta(mfcc, order=2), axis=1)
            mfcc_skew = skew(mfcc, axis=1)
            mfcc_kurtosis = kurtosis(mfcc, axis=1)
            
            # Mel Spectrogram Features
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_mean = np.mean(mel_spec, axis=1)
            mel_std = np.std(mel_spec, axis=1)
            
            # Spectral Features
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            
            # Chroma Features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            chroma_std = np.std(chroma, axis=1)
            
            # Tonnetz Features
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            tonnetz_mean = np.mean(tonnetz, axis=1)
            tonnetz_std = np.std(tonnetz, axis=1)
            
            # Rhythm Features
            tempo_result = librosa.beat.beat_track(y=y, sr=sr, start_bpm=120)
            tempo = float(tempo_result[0]) if isinstance(tempo_result, tuple) else float(tempo_result)
            
            # RMS Energy
            rms = librosa.feature.rms(y=y)
            
            # Normalize spectral features
            spectral_features = np.array([
                np.mean(spectral_centroid) / 10000.0,
                np.std(spectral_centroid) / 10000.0,
                np.mean(spectral_rolloff) / 10000.0,
                np.std(spectral_rolloff) / 10000.0,
                np.mean(spectral_bandwidth) / 10000.0,
                np.std(spectral_bandwidth) / 10000.0,
                np.mean(zero_crossing_rate) * 100.0,
                np.std(zero_crossing_rate) * 100.0,
                tempo / 200.0,
                np.mean(rms) * 10.0,
                np.std(rms) * 10.0
            ], dtype=np.float32)
            
            # Combine all features with appropriate weights
            embedding = np.concatenate([
                mfcc_mean.flatten() * 3.0,
                mfcc_std.flatten() * 2.0,
                mfcc_delta.flatten() * 2.5,
                mfcc_delta2.flatten() * 1.5,
                mfcc_skew.flatten() * 1.0,
                mfcc_kurtosis.flatten() * 1.0,
                mel_mean.flatten() * 0.3,
                mel_std.flatten() * 0.2,
                chroma_mean.flatten() * 1.2,
                chroma_std.flatten() * 0.8,
                np.mean(spectral_contrast, axis=1).flatten() * 1.5,
                np.std(spectral_contrast, axis=1).flatten() * 1.0,
                tonnetz_mean.flatten() * 1.0,
                tonnetz_std.flatten() * 0.8,
                spectral_features
            ])
            
            # L2 normalization
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            else:
                print(f"Warning: Zero norm for {audio_path}")
                return None
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            return None
    
    def add_audio_file(self, audio_path: str, metadata: Dict = None) -> bool:
        """
        Add a single audio file to the database.
        
        Args:
            audio_path: Path to the audio file
            metadata: Optional metadata dictionary
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(audio_path):
            print(f"File not found: {audio_path}")
            return False
        
        embedding = self.extract_audio_features(audio_path)
        if embedding is None:
            return False
        
        file_name = os.path.basename(audio_path)
        file_metadata = {
            "path": audio_path,
            "filename": file_name,
            "size": os.path.getsize(audio_path)
        }
        
        if metadata:
            file_metadata.update(metadata)
        
        file_id = f"audio_{hash(audio_path) % 10**10}"
        
        try:
            self.collection.add(
                ids=[file_id],
                embeddings=[embedding.tolist()],
                metadatas=[file_metadata]
            )
            print(f"✓ Added: {file_name}")
            return True
        except Exception as e:
            print(f"Error adding {file_name}: {str(e)}")
            return False
    
    def add_audio_directory(self, directory_path: str, extensions: List[str] = None) -> int:
        """
        Add all audio files from a directory to the database.
        
        Args:
            directory_path: Path to directory containing audio files
            extensions: List of audio file extensions to include
            
        Returns:
            Number of files successfully added
        """
        if extensions is None:
            extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac']
        
        if not os.path.exists(directory_path):
            print(f"Directory not found: {directory_path}")
            return 0
        
        audio_files = []
        for ext in extensions:
            audio_files.extend(Path(directory_path).rglob(f"*{ext}"))
        
        print(f"\nFound {len(audio_files)} audio files in {directory_path}")
        
        success_count = 0
        for audio_file in audio_files:
            if self.add_audio_file(str(audio_file)):
                success_count += 1
        
        print(f"\nSuccessfully added {success_count}/{len(audio_files)} files")
        return success_count
    
    def search_similar(self, query_audio_path: str, top_k: int = 5) -> List[Tuple[str, float, float]]:
        """
        Search for similar audio files using cosine similarity.
        
        Args:
            query_audio_path: Path to the query audio file
            top_k: Number of similar results to return
            
        Returns:
            List of tuples (audio_path, similarity_score, distance)
        """
        if not os.path.exists(query_audio_path):
            print(f"Query file not found: {query_audio_path}")
            return []
        
        query_embedding = self.extract_audio_features(query_audio_path)
        if query_embedding is None:
            return []
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self.collection.count())
        )
        
        similar_files = []
        if results and results['metadatas'] and results['distances']:
            for metadata, distance in zip(results['metadatas'][0], results['distances'][0]):
                cosine_similarity = 1 - distance
                similarity_score = max(0, min(100, cosine_similarity * 100))
                similar_files.append((metadata['path'], similarity_score, distance))
        
        return similar_files
    
    def print_search_results(self, query_path: str, results: List[Tuple[str, float, float]]):
        """Print search results in a formatted way."""
        print(f"\n{'='*70}")
        print(f"Query Audio: {os.path.basename(query_path)}")
        print(f"{'='*70}")
        
        if not results:
            print("No similar audio files found.")
            return
        
        print(f"\n🎯 Best Match:")
        print(f"{'-'*70}")
        
        path, similarity, distance = results[0]
        print(f"✓ {os.path.basename(path)}")
        print(f"   Similarity Score: {similarity:.2f}%")
        print(f"   Cosine Distance: {distance:.6f}")
        print(f"   Path: {path}")
        
        if similarity >= 80:
            interpretation = "Very High Similarity - Likely same speaker/source"
        elif similarity >= 60:
            interpretation = "High Similarity - Very similar characteristics"
        elif similarity >= 40:
            interpretation = "Moderate Similarity - Some common features"
        else:
            interpretation = "Low Similarity - Different characteristics"
        
        print(f"   Interpretation: {interpretation}")
        print()
    
    def get_database_info(self) -> Dict:
        """Get information about the current database."""
        return {
            "collection_name": self.collection_name,
            "total_files": self.collection.count(),
            "persist_directory": self.persist_directory,
            "distance_metric": "cosine"
        }
    
    def clear_database(self):
        """Clear all entries from the database."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "Audio file embeddings for similarity search",
                "hnsw:space": "cosine"
            }
        )
        print(f"Database cleared. Collection: {self.collection_name}")


if __name__ == "__main__":
    audio_search = AudioSimilaritySearch(
        collection_name="my_audio_collection",
        persist_directory="./audio_database"
    )
    
    print("\n=== Adding Audio Files from Directory ===")
    audio_search.add_audio_directory("./input")
    
    print("\n=== Searching for Similar Audio ===")
    query_audio = "./query/query_audio.mp3"
    #query_audio = "./query/query_audio.m4a"
    similar_files = audio_search.search_similar(query_audio, top_k=10)
    audio_search.print_search_results(query_audio, similar_files)
    
    print("\n=== Database Information ===")
    info = audio_search.get_database_info()
    print(f"Collection: {info['collection_name']}")
    print(f"Total Audio Files: {info['total_files']}")
    print(f"Storage Location: {info['persist_directory']}")
    print(f"Distance Metric: {info['distance_metric']}")