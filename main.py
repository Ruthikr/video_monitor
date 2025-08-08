#!/usr/bin/env python3

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import librosa
import torch
import torchvision
from scipy import signal
from scipy.spatial.distance import cosine
import json
import os
import time
import logging
import tempfile
import shutil
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
import uvicorn

warnings.filterwarnings('ignore')

# Initialize FastAPI app
app = FastAPI(
    title="Video Monitoring System API",
    description="API for detecting integrity issues in video content",
    version="1.0.0"
)

@dataclass
class DetectionConfig:
    """Configuration class for all detection parameters"""
    
    # Video Processing
    target_fps: int = 24
    frame_skip: int = 1
    max_video_length: int = 1800  # 30 minutes in seconds
    
    # Face Detection
    face_confidence_threshold: float = 0.7
    min_face_size: int = 60  # pixels
    face_temporal_filter: int = 5  # consecutive frames
    max_face_count: int = 10  # reasonable upper limit
    
    # Voice Detection
    audio_window_size: float = 2.0  # seconds
    audio_overlap: float = 0.5 # seconds
    voice_confidence_threshold: float = 0.8
    min_speech_duration: float = 1.0  # seconds
    
    # Gaze Tracking
    gaze_center_threshold: float = 15.0  # degrees
    gaze_acceptable_threshold: float = 30.0  # degrees
    gaze_temporal_window: float = 1.0  # seconds
    gaze_deviation_duration: float = 3.0  # seconds
    
    # Lip Sync
    sync_window_size: float = 5.0  # seconds
    sync_tolerance: float = 200.0  # milliseconds
    mouth_movement_threshold: float = 0.02
    audio_energy_threshold: float = 0.01
    
    # Performance
    batch_size: int = 32
    num_threads: int = 4
    gpu_acceleration: bool = True

# Event types enumeration
class EventType:
    MULTIPLE_FACES = "MULTIPLE_FACES_DETECTED"
    MULTIPLE_VOICES = "MULTIPLE_VOICES_DETECTED"
    FACE_OUT_OF_VIEW = "FACE_OUT_OF_VIEW"
    EYE_GAZE_DEVIATION = "EYE_GAZE_DEVIATION"
    LIP_SYNC_ISSUE = "LIP_SYNC_ISSUE"
    FACE_RETURNED = "FACE_RETURNED_TO_VIEW"
    GAZE_NORMALIZED = "GAZE_RETURNED_TO_CENTER"

@dataclass
class DetectionEvent:
    """Data class for detection events"""
    timestamp: float
    event_type: str
    confidence: float
    details: Dict[str, Any]
    duration: Optional[float] = None

class VideoProcessor:
    """Core video processing utilities"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        
    def load_video(self, video_path: str) -> Tuple[cv2.VideoCapture, Dict[str, Any]]:
        """Load video and extract metadata"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        metadata = {
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration,
            'width': width,
            'height': height,
            'total_frames': frame_count
        }
        
        return cap, metadata
    
    def extract_audio(self, video_path: str) -> Tuple[np.ndarray, int]:
        """Extract audio from video file"""
        try:
            audio, sr = librosa.load(video_path, sr=16000)
            return audio, sr
        except Exception as e:
            logging.error(f"Failed to extract audio: {e}")
            return np.array([]), 16000
    
    def frame_generator(self, cap: cv2.VideoCapture, skip_frames: int = 1):
        """Generate frames from video with optional skipping"""
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % skip_frames == 0:
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                yield frame, timestamp, frame_idx
            
            frame_idx += 1
    
    def format_timestamp(self, seconds: float) -> str:
        """Format timestamp as MM:SS.mmm"""
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:02d}:{secs:06.3f}"

class MultipleFaceDetector:
    """Detects multiple faces in video frames"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # Use full-range model for better accuracy
            min_detection_confidence=config.face_confidence_threshold
        )
        
        # Temporal filtering
        self.face_count_history = []
        self.last_event_time = 0
        
    def detect_faces_mediapipe(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        faces = []
        if results.detections:
            h, w, _ = frame.shape
            
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                confidence = detection.score[0]
                
                # Convert to absolute coordinates
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Filter by minimum face size
                if width >= self.config.min_face_size and height >= self.config.min_face_size:
                    faces.append({
                        'bbox': (x, y, width, height),
                        'confidence': confidence,
                        'size': width * height
                    })
        
        return faces
    
    def process_frame(self, frame: np.ndarray, timestamp: float) -> List[DetectionEvent]:
        """Process single frame for multiple faces"""
        events = []
        
        faces = self.detect_faces_mediapipe(frame)
        face_count = len(faces)
        
        # Apply temporal filtering
        self.face_count_history.append(face_count)
        if len(self.face_count_history) > self.config.face_temporal_filter:
            self.face_count_history.pop(0)
        
        # Check for stable multiple faces detection
        if len(self.face_count_history) == self.config.face_temporal_filter:
            stable_count = np.median(self.face_count_history)
            
            # Multiple faces detected
            if stable_count > 1 and (timestamp - self.last_event_time) > 2.0:
                avg_confidence = np.mean([f['confidence'] for f in faces]) if faces else 0
                
                events.append(DetectionEvent(
                    timestamp=timestamp,
                    event_type=EventType.MULTIPLE_FACES,
                    confidence=avg_confidence,
                    details={
                        'face_count': int(stable_count),
                        'faces': faces,
                        'frame_resolution': frame.shape[:2]
                    }
                ))
                self.last_event_time = timestamp
            
            # Face out of view detection
            elif stable_count == 0 and (timestamp - self.last_event_time) > 2.0:
                events.append(DetectionEvent(
                    timestamp=timestamp,
                    event_type=EventType.FACE_OUT_OF_VIEW,
                    confidence=0.95,
                    details={
                        'face_count': 0,
                        'duration_start': timestamp
                    }
                ))
                self.last_event_time = timestamp
        
        return events

class MultipleVoiceDetector:
    """Detects multiple distinct voices in audio"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.sample_rate = 16000
        
        # Voice embeddings for speaker comparison
        self.speaker_embeddings = []
        self.voice_threshold = 0.7  # Similarity threshold for same speaker
        
        # Audio processing parameters
        self.hop_length = int(config.audio_window_size * self.sample_rate * config.audio_overlap)
        self.win_length = int(config.audio_window_size * self.sample_rate)
        
    def extract_voice_features(self, audio_segment: np.ndarray) -> Dict[str, Any]:
        """Extract voice features from audio segment"""
        if len(audio_segment) < 1000:  # Too short
            return {}
        
        try:
            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio_segment, sr=self.sample_rate, n_mfcc=13)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_segment, sr=self.sample_rate)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_segment, sr=self.sample_rate)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_segment)
            
            # Pitch estimation
            pitches, magnitudes = librosa.piptrack(y=audio_segment, sr=self.sample_rate)
            pitch = np.mean(pitches[pitches > 0]) if len(pitches[pitches > 0]) > 0 else 0
            
            features = {
                'mfcc_mean': np.mean(mfccs, axis=1),
                'mfcc_std': np.std(mfccs, axis=1),
                'spectral_centroid': np.mean(spectral_centroids),
                'spectral_rolloff': np.mean(spectral_rolloff),
                'zero_crossing_rate': np.mean(zero_crossing_rate),
                'pitch': pitch,
                'energy': np.mean(audio_segment**2)
            }
            
            return features
        except Exception:
            return {}
    
    def is_speech(self, audio_segment: np.ndarray) -> bool:
        """Determine if audio segment contains speech"""
        if len(audio_segment) == 0:
            return False
        
        # Simple energy-based voice activity detection
        energy = np.mean(audio_segment**2)
        
        # Zero crossing rate (speech typically has moderate ZCR)
        zcr = librosa.feature.zero_crossing_rate(audio_segment)[0]
        mean_zcr = np.mean(zcr)
        
        # Spectral centroid (speech typically has lower centroid than noise)
        spec_cent = librosa.feature.spectral_centroid(y=audio_segment, sr=self.sample_rate)
        mean_spec_cent = np.mean(spec_cent)
        
        # Heuristic thresholds for speech detection
        is_speech = (
            energy > self.config.audio_energy_threshold and
            0.01 < mean_zcr < 0.25 and
            1000 < mean_spec_cent < 4000
        )
        
        return is_speech
    
    def compare_voices(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """Compare two voice feature sets and return similarity score"""
        if not features1 or not features2:
            return 0.0
        
        # Compare MFCC features (primary voice characteristics)
        mfcc_similarity = 1 - cosine(features1['mfcc_mean'], features2['mfcc_mean'])
        
        # Compare pitch (fundamental frequency)
        pitch_diff = abs(features1['pitch'] - features2['pitch'])
        pitch_similarity = max(0, 1 - (pitch_diff / 200))  # Normalize by 200 Hz
        
        # Compare spectral features
        spec_cent_diff = abs(features1['spectral_centroid'] - features2['spectral_centroid'])
        spec_similarity = max(0, 1 - (spec_cent_diff / 2000))  # Normalize by 2000 Hz
        
        # Weighted combination
        similarity = (0.6 * mfcc_similarity + 0.25 * pitch_similarity + 0.15 * spec_similarity)
        
        return max(0, min(1, similarity))  # Clamp to [0, 1]
    
    def process_audio_window(self, audio: np.ndarray, start_time: float) -> List[DetectionEvent]:
        """Process audio window for multiple voice detection"""
        events = []
        
        # Check if this segment contains speech
        if not self.is_speech(audio):
            return events
        
        # Extract voice features
        features = self.extract_voice_features(audio)
        if not features:
            return events
        
        # Compare with existing speaker embeddings
        is_new_speaker = True
        max_similarity = 0
        
        for existing_features in self.speaker_embeddings:
            similarity = self.compare_voices(features, existing_features)
            max_similarity = max(max_similarity, similarity)
            
            if similarity > self.voice_threshold:
                is_new_speaker = False
                break
        
        # Add new speaker if sufficiently different
        if is_new_speaker and len(self.speaker_embeddings) < 10:  # Limit speaker count
            self.speaker_embeddings.append(features)
        
        # Report multiple speakers if detected
        num_speakers = len(self.speaker_embeddings)
        if num_speakers > 1:
            events.append(DetectionEvent(
                timestamp=start_time,
                event_type=EventType.MULTIPLE_VOICES,
                confidence=min(0.95, 0.5 + (num_speakers * 0.15)),
                details={
                    'speaker_count': num_speakers,
                    'current_speaker_similarity': max_similarity,
                    'audio_energy': features['energy'],
                    'window_duration': len(audio) / self.sample_rate
                }
            ))
        
        return events
    
    def process_audio(self, audio: np.ndarray) -> List[DetectionEvent]:
        """Process entire audio track for voice detection"""
        events = []
        
        if len(audio) == 0:
            return events
        
        window_samples = int(self.config.audio_window_size * self.sample_rate)
        hop_samples = int(window_samples * (1 - self.config.audio_overlap))
        
        for start_sample in range(0, len(audio) - window_samples, hop_samples):
            end_sample = start_sample + window_samples
            audio_window = audio[start_sample:end_sample]
            start_time = start_sample / self.sample_rate
            
            window_events = self.process_audio_window(audio_window, start_time)
            events.extend(window_events)
        
        return events

class EyeGazeTracker:
    """Tracks eye gaze direction and detects deviations"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        
        # Initialize MediaPipe Face Mesh for detailed landmarks
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye landmark indices for MediaPipe Face Mesh (468 landmarks)
        self.LEFT_EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Iris landmarks (available with refine_landmarks=True)
        self.LEFT_IRIS = [468, 469, 470, 471, 472]
        self.RIGHT_IRIS = [473, 474, 475, 476, 477]
        
        # Gaze history for temporal smoothing
        self.gaze_history = []
        self.last_deviation_time = 0
        
    def get_eye_landmarks(self, landmarks, eye_indices: List[int]) -> np.ndarray:
        """Extract eye landmarks as numpy array"""
        points = []
        for idx in eye_indices:
            point = landmarks.landmark[idx]
            points.append([point.x, point.y, point.z])
        return np.array(points)
    
    def get_iris_center(self, landmarks, iris_indices: List[int]) -> np.ndarray:
        """Get iris center from iris landmarks"""
        iris_points = []
        for idx in iris_indices:
            point = landmarks.landmark[idx]
            iris_points.append([point.x, point.y, point.z])
        
        iris_points = np.array(iris_points)
        return np.mean(iris_points, axis=0)
    
    def calculate_gaze_direction(self, eye_landmarks: np.ndarray, iris_center: np.ndarray) -> Tuple[float, float]:
        """Calculate gaze direction angles"""
        # Get eye corners
        eye_left = eye_landmarks[0]  # Left corner
        eye_right = eye_landmarks[8]  # Right corner
        eye_center = (eye_left + eye_right) / 2
        
        # Calculate gaze vector
        gaze_vector = iris_center - eye_center
        
        # Convert to angles (in degrees)
        horizontal_angle = np.arctan2(gaze_vector[0], gaze_vector[2]) * 180 / np.pi
        vertical_angle = np.arctan2(gaze_vector[1], gaze_vector[2]) * 180 / np.pi
        
        return horizontal_angle, vertical_angle
    
    def classify_gaze_zone(self, h_angle: float, v_angle: float) -> str:
        """Classify gaze into zones"""
        abs_h = abs(h_angle)
        abs_v = abs(v_angle)
        
        if abs_h < self.config.gaze_center_threshold and abs_v < self.config.gaze_center_threshold:
            return "CENTER"
        elif abs_h > self.config.gaze_acceptable_threshold or abs_v > self.config.gaze_acceptable_threshold:
            return "OFF_SCREEN"
        else:
            return "ACCEPTABLE"
    
    def process_frame(self, frame: np.ndarray, timestamp: float) -> List[DetectionEvent]:
        """Process frame for gaze tracking"""
        events = []
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return events
        
        # Process first face only
        face_landmarks = results.multi_face_landmarks[0]
        
        try:
            # Get eye landmarks
            left_eye = self.get_eye_landmarks(face_landmarks, self.LEFT_EYE_LANDMARKS)
            right_eye = self.get_eye_landmarks(face_landmarks, self.RIGHT_EYE_LANDMARKS)
            
            # Get iris centers
            left_iris = self.get_iris_center(face_landmarks, self.LEFT_IRIS)
            right_iris = self.get_iris_center(face_landmarks, self.RIGHT_IRIS)
            
            # Calculate gaze direction for both eyes
            left_h, left_v = self.calculate_gaze_direction(left_eye, left_iris)
            right_h, right_v = self.calculate_gaze_direction(right_eye, right_iris)
            
            # Average both eyes for final gaze direction
            avg_h = (left_h + right_h) / 2
            avg_v = (left_v + right_v) / 2
            
            # Apply temporal smoothing
            self.gaze_history.append((avg_h, avg_v, timestamp))
            
            # Keep only recent history
            window_start = timestamp - self.config.gaze_temporal_window
            self.gaze_history = [(h, v, t) for h, v, t in self.gaze_history if t >= window_start]
            
            # Calculate smoothed gaze
            if len(self.gaze_history) >= 3:
                recent_h = [h for h, v, t in self.gaze_history[-3:]]
                recent_v = [v for h, v, t in self.gaze_history[-3:]]
                smooth_h = np.mean(recent_h)
                smooth_v = np.mean(recent_v)
                
                # Classify gaze zone
                gaze_zone = self.classify_gaze_zone(smooth_h, smooth_v)
                
                # Check for sustained deviation
                if gaze_zone == "OFF_SCREEN":
                    if timestamp - self.last_deviation_time > self.config.gaze_deviation_duration:
                        direction = "Left" if smooth_h < -self.config.gaze_acceptable_threshold else \
                                   "Right" if smooth_h > self.config.gaze_acceptable_threshold else \
                                   "Up" if smooth_v < -self.config.gaze_acceptable_threshold else \
                                   "Down" if smooth_v > self.config.gaze_acceptable_threshold else ""
                        
                        confidence = min(0.95, 0.6 + (abs(smooth_h) + abs(smooth_v)) / 100)
                        
                        events.append(DetectionEvent(
                            timestamp=timestamp,
                            event_type=EventType.EYE_GAZE_DEVIATION,
                            confidence=confidence,
                            details={
                                'horizontal_angle': round(smooth_h, 2),
                                'vertical_angle': round(smooth_v, 2),
                                'direction': direction,
                                'gaze_zone': gaze_zone,
                                'left_eye_angles': (round(left_h, 2), round(left_v, 2)),
                                'right_eye_angles': (round(right_h, 2), round(right_v, 2))
                            }
                        ))
                        self.last_deviation_time = timestamp
        
        except Exception as e:
            logging.warning(f"Gaze tracking error at {timestamp:.2f}s: {e}")
        
        return events

class LipSyncAnalyzer:
    """Analyzes lip sync between visual mouth movement and audio"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        
        # Initialize MediaPipe Face Mesh for mouth landmarks
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Mouth landmark indices (outer lip contour)
        self.MOUTH_LANDMARKS = [
            61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318,
            13, 82, 81, 80, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324
        ]
        
    def get_mouth_landmarks(self, landmarks) -> np.ndarray:
        """Extract mouth landmarks"""
        points = []
        for idx in self.MOUTH_LANDMARKS:
            point = landmarks.landmark[idx]
            points.append([point.x, point.y])
        return np.array(points)
    
    def calculate_mouth_opening(self, landmarks) -> float:
        """Calculate mouth opening ratio"""
        # Get key mouth points
        upper_lip = landmarks.landmark[13]  # Upper lip center
        lower_lip = landmarks.landmark[14]  # Lower lip center
        left_corner = landmarks.landmark[61]  # Left mouth corner
        right_corner = landmarks.landmark[291]  # Right mouth corner
        
        # Calculate vertical opening (height)
        mouth_height = abs(upper_lip.y - lower_lip.y)
        
        # Calculate horizontal width
        mouth_width = abs(left_corner.x - right_corner.x)
        
        # Mouth opening ratio (height/width)
        if mouth_width > 0:
            opening_ratio = mouth_height / mouth_width
        else:
            opening_ratio = 0
        
        return opening_ratio
    
    def calculate_mouth_movement_velocity(self, current_landmarks: np.ndarray, 
                                       previous_landmarks: np.ndarray) -> float:
        """Calculate mouth movement velocity between frames"""
        if previous_landmarks is None or len(previous_landmarks) == 0:
            return 0.0
        
        # Calculate movement for each landmark
        movements = np.sqrt(np.sum((current_landmarks - previous_landmarks) ** 2, axis=1))
        
        # Return average movement
        return np.mean(movements)
    
    def extract_audio_envelope(self, audio_segment: np.ndarray, sr: int) -> np.ndarray:
        """Extract audio envelope (speech energy over time)"""
        if len(audio_segment) == 0:
            return np.array([])
        
        # Apply pre-emphasis filter
        pre_emphasis = 0.97
        emphasized_audio = np.append(audio_segment[0], audio_segment[1:] - pre_emphasis * audio_segment[:-1])
        
        # Calculate short-time energy
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)   # 10ms hop
        
        # Windowed energy calculation
        energy = []
        for i in range(0, len(emphasized_audio) - frame_length, hop_length):
            frame = emphasized_audio[i:i + frame_length]
            frame_energy = np.sum(frame ** 2)
            energy.append(frame_energy)
        
        return np.array(energy)
    
    def correlate_audio_visual(self, mouth_movements: List[float], 
                             audio_envelope: np.ndarray, 
                             video_fps: float, 
                             audio_sr: int) -> Tuple[float, float]:
        """Calculate correlation between mouth movement and audio"""
        if len(mouth_movements) == 0 or len(audio_envelope) == 0:
            return 0.0, 0.0
        
        # Resample to common time base
        common_length = min(len(mouth_movements), len(audio_envelope))
        
        if common_length < 3:
            return 0.0, 0.0
        
        mouth_resampled = np.interp(np.linspace(0, 1, common_length), 
                                  np.linspace(0, 1, len(mouth_movements)), 
                                  mouth_movements)
        audio_resampled = np.interp(np.linspace(0, 1, common_length),
                                  np.linspace(0, 1, len(audio_envelope)),
                                  audio_envelope)
        
        # Normalize
        if np.std(mouth_resampled) > 0:
            mouth_normalized = (mouth_resampled - np.mean(mouth_resampled)) / np.std(mouth_resampled)
        else:
            mouth_normalized = mouth_resampled
            
        if np.std(audio_resampled) > 0:
            audio_normalized = (audio_resampled - np.mean(audio_resampled)) / np.std(audio_resampled)
        else:
            audio_normalized = audio_resampled
        
        # Cross-correlation to find optimal delay
        correlation = np.correlate(mouth_normalized, audio_normalized, mode='full')
        
        # Find peak correlation and delay
        peak_idx = np.argmax(np.abs(correlation))
        max_correlation = correlation[peak_idx]
        delay_samples = peak_idx - (len(audio_normalized) - 1)
        
        # Convert delay to milliseconds
        video_duration = len(mouth_movements) / video_fps
        delay_ms = (delay_samples / common_length) * (video_duration * 1000)
        
        return abs(max_correlation), delay_ms
    
    def analyze_sync_window(self, video_frames: List[np.ndarray], 
                          audio_segment: np.ndarray, 
                          start_time: float, 
                          fps: float, 
                          audio_sr: int) -> List[DetectionEvent]:
        """Analyze lip sync for a time window"""
        events = []
        
        if len(video_frames) == 0 or len(audio_segment) == 0:
            return events
        
        mouth_movements = []
        mouth_openings = []
        previous_mouth_landmarks = None
        
        # Process video frames
        for frame in video_frames:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                # Get mouth landmarks and opening
                mouth_landmarks = self.get_mouth_landmarks(face_landmarks)
                mouth_opening = self.calculate_mouth_opening(face_landmarks)
                mouth_openings.append(mouth_opening)
                
                # Calculate movement velocity
                if previous_mouth_landmarks is not None:
                    movement = self.calculate_mouth_movement_velocity(
                        mouth_landmarks, previous_mouth_landmarks
                    )
                    mouth_movements.append(movement)
                
                previous_mouth_landmarks = mouth_landmarks
            else:
                mouth_openings.append(0)
                mouth_movements.append(0)
        
        # Extract audio envelope
        audio_envelope = self.extract_audio_envelope(audio_segment, audio_sr)
        
        # Analyze correlation
        correlation, delay_ms = self.correlate_audio_visual(
            mouth_movements, audio_envelope, fps, audio_sr
        )
        
        # Check for sync issues
        sync_tolerance = self.config.sync_tolerance
        
        if abs(delay_ms) > sync_tolerance or correlation < 0.3:
            events.append(DetectionEvent(
                timestamp=start_time,
                event_type=EventType.LIP_SYNC_ISSUE,
                confidence=min(0.95, 0.5 + (abs(delay_ms) / 1000)),
                details={
                    'delay_ms': round(delay_ms, 2),
                    'correlation': round(correlation, 4),
                    'sync_tolerance': sync_tolerance,
                    'mouth_movement_avg': round(np.mean(mouth_movements) if mouth_movements else 0, 4),
                    'window_duration': len(video_frames) / fps
                }
            ))
        
        return events

class VideoMonitoringSystem:
    """Main system that orchestrates all detection modules"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.processor = VideoProcessor(config)
        
        # Initialize detection modules
        self.face_detector = MultipleFaceDetector(config)
        self.voice_detector = MultipleVoiceDetector(config)
        self.gaze_tracker = EyeGazeTracker(config)
        self.lipsync_analyzer = LipSyncAnalyzer(config)
        
        # Event storage
        self.all_events = []
        
    def process_video(self, video_path: str) -> Dict[str, Any]:
        """Process entire video and return analysis results"""
        
        # Load video and extract metadata
        cap, metadata = self.processor.load_video(video_path)
        audio, audio_sr = self.processor.extract_audio(video_path)
        
        # Process video frames
        frame_events = []
        processed_frames = 0
        total_frames = int(metadata['frame_count'] / self.config.frame_skip)
        
        # Buffers for lip sync analysis
        lipsync_frame_buffer = []
        lipsync_audio_buffer = []
        lipsync_start_time = 0
        
        for frame, timestamp, frame_idx in self.processor.frame_generator(cap, self.config.frame_skip):
            try:
                # Face detection and gaze tracking
                face_events = self.face_detector.process_frame(frame, timestamp)
                gaze_events = self.gaze_tracker.process_frame(frame, timestamp)
                
                frame_events.extend(face_events)
                frame_events.extend(gaze_events)
                
                # Collect frames for lip sync analysis
                lipsync_frame_buffer.append(frame)
                
                # Extract corresponding audio segment
                audio_start_idx = int(timestamp * audio_sr)
                audio_end_idx = int((timestamp + 1/metadata['fps']) * audio_sr)
                if audio_end_idx < len(audio):
                    lipsync_audio_buffer.extend(audio[audio_start_idx:audio_end_idx])
                
                # Process lip sync window when buffer is full
                buffer_duration = len(lipsync_frame_buffer) / metadata['fps']
                if buffer_duration >= self.config.sync_window_size:
                    audio_segment = np.array(lipsync_audio_buffer)
                    
                    lipsync_events = self.lipsync_analyzer.analyze_sync_window(
                        lipsync_frame_buffer, 
                        audio_segment,
                        lipsync_start_time,
                        metadata['fps'],
                        audio_sr
                    )
                    
                    frame_events.extend(lipsync_events)
                    
                    # Clear buffers
                    lipsync_frame_buffer = []
                    lipsync_audio_buffer = []
                    lipsync_start_time = timestamp
                
                processed_frames += 1
                
            except Exception as e:
                logging.error(f"Error processing frame at {timestamp:.2f}s: {e}")
                continue
        
        # Process audio for voice detection
        voice_events = self.voice_detector.process_audio(audio)
        
        # Combine all events
        all_events = frame_events + voice_events
        all_events.sort(key=lambda x: x.timestamp)
        
        # Clean up
        cap.release()
        
        # Generate analysis report
        report = self.generate_report(all_events, metadata)
        
        return {
            'events': all_events,
            'metadata': metadata,
            'report': report,
            'video_path': video_path
        }
    
    def generate_report(self, events: List[DetectionEvent], metadata: Dict) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        
        # Group events by type
        event_counts = {}
        event_durations = {}
        
        for event in events:
            event_type = event.event_type
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            if event.duration:
                if event_type not in event_durations:
                    event_durations[event_type] = []
                event_durations[event_type].append(event.duration)
        
        # Calculate statistics
        total_events = len(events)
        video_duration = metadata['duration']
        
        # Event density (events per minute)
        event_density = (total_events / video_duration) * 60 if video_duration > 0 else 0
        
        # Confidence statistics
        confidences = [event.confidence for event in events]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        # Generate summary
        report = {
            'total_events': total_events,
            'event_counts': event_counts,
            'event_density_per_minute': round(event_density, 2),
            'average_confidence': round(avg_confidence, 3),
            'video_duration_seconds': round(video_duration, 2),
            'analysis_summary': {
                'integrity_issues_detected': total_events > 0,
                'multiple_faces_count': event_counts.get(EventType.MULTIPLE_FACES, 0),
                'multiple_voices_count': event_counts.get(EventType.MULTIPLE_VOICES, 0),
                'face_out_of_view_count': event_counts.get(EventType.FACE_OUT_OF_VIEW, 0),
                'gaze_deviation_count': event_counts.get(EventType.EYE_GAZE_DEVIATION, 0),
                'lip_sync_issues_count': event_counts.get(EventType.LIP_SYNC_ISSUE, 0)
            },
            'processing_metadata': {
                'video_resolution': f"{metadata['width']}x{metadata['height']}",
                'video_fps': metadata['fps'],
                'total_frames_processed': int(metadata['frame_count'] / self.config.frame_skip),
                'processing_fps': self.config.target_fps
            }
        }
        
        return report

def format_timestamp(seconds: float) -> str:
    """Format timestamp as MM:SS.mmm"""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:06.3f}"

def format_event_details(details: Dict[str, Any]) -> str:
    """Format event details for display"""
    if not details:
        return "No additional details"
    
    key_details = []
    for key, value in details.items():
        if key in ['face_count', 'speaker_count', 'direction', 'delay_ms', 'correlation']:
            if isinstance(value, float):
                key_details.append(f"{key}: {value:.2f}")
            else:
                key_details.append(f"{key}: {value}")
    
    return " | ".join(key_details) if key_details else "Additional details available"

def export_results_to_log(results: Dict[str, Any]) -> str:
    """Export results to log string"""
    events = results['events']
    report = results['report']
    metadata = results['metadata']
    
    log_content = []
    
    # Write header
    log_content.append("=" * 80)
    log_content.append("VIDEO MONITORING SYSTEM - ANALYSIS REPORT")
    log_content.append("=" * 80)
    log_content.append(f"Analysis Timestamp: {datetime.now().isoformat()}")
    log_content.append(f"Video File: {results['video_path']}")
    log_content.append(f"Video Duration: {report['video_duration_seconds']:.1f} seconds")
    log_content.append(f"Video Resolution: {report['processing_metadata']['video_resolution']}")
    log_content.append(f"Video FPS: {metadata['fps']:.1f}")
    log_content.append("")
    
    # Summary statistics
    log_content.append("ANALYSIS SUMMARY")
    log_content.append("-" * 40)
    log_content.append(f"Total Events Detected: {report['total_events']}")
    log_content.append(f"Event Density: {report['event_density_per_minute']:.1f} events/minute")
    log_content.append(f"Average Confidence: {report['average_confidence']:.3f}")
    log_content.append("")
    
    # Event breakdown
    log_content.append("EVENT BREAKDOWN")
    log_content.append("-" * 40)
    summary = report['analysis_summary']
    log_content.append(f"Multiple Faces: {summary['multiple_faces_count']}")
    log_content.append(f"Multiple Voices: {summary['multiple_voices_count']}")
    log_content.append(f"Face Out of View: {summary['face_out_of_view_count']}")
    log_content.append(f"Gaze Deviations: {summary['gaze_deviation_count']}")
    log_content.append(f"Lip Sync Issues: {summary['lip_sync_issues_count']}")
    log_content.append("")
    
    # Detailed event log
    if events:
        log_content.append("DETAILED EVENT LOG")
        log_content.append("-" * 80)
        log_content.append(f"{'Time':<12} {'Event Type':<25} {'Confidence':<12} {'Details'}")
        log_content.append("-" * 80)
        
        for event in events:
            timestamp_str = format_timestamp(event.timestamp)
            details_str = format_event_details(event.details)
            log_content.append(f"{timestamp_str:<12} {event.event_type:<25} {event.confidence:.3f}        {details_str}")
    
    log_content.append("")
    log_content.append("=" * 80)
    log_content.append("END OF REPORT")
    log_content.append("=" * 80)
    
    return "\n".join(log_content)

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Video Monitoring System API",
        "version": "1.0.0",
        "description": "Upload videos to detect integrity issues",
        "endpoints": {
            "/analyze": "POST - Upload video for analysis",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.post("/analyze")
async def analyze_video(
    background_tasks: BackgroundTasks,
    video_file: UploadFile = File(...),
    config_preset: str = "default",
    return_format: str = "log"
):
    """
    Analyze uploaded video for integrity issues
    
    - **video_file**: Video file to analyze (mp4, avi, mov, etc.)
    - **config_preset**: Configuration preset ('default', 'sensitive', 'performance')
    - **return_format**: Response format ('log', 'json', 'csv')
    """
    
    # Validate file type
    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    file_extension = os.path.splitext(video_file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed formats: {', '.join(allowed_extensions)}"
        )
    
    # Create temporary file
    temp_dir = tempfile.mkdtemp()
    temp_video_path = os.path.join(temp_dir, f"input_video{file_extension}")
    temp_log_path = os.path.join(temp_dir, "analysis_log.txt")
    temp_csv_path = os.path.join(temp_dir, "analysis_results.csv")
    
    try:
        # Save uploaded file
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)
        
        # Configure detection system
        if config_preset == "sensitive":
            config = DetectionConfig()
            config.face_confidence_threshold = 0.5
            config.voice_confidence_threshold = 0.6
            config.gaze_acceptable_threshold = 20.0
            config.sync_tolerance = 150.0
        elif config_preset == "performance":
            config = DetectionConfig()
            config.target_fps = 10
            config.frame_skip = 3
            config.face_confidence_threshold = 0.8
            config.voice_confidence_threshold = 0.9
        else:
            config = DetectionConfig()
        
        # Initialize and run analysis
        system = VideoMonitoringSystem(config)
        results = system.process_video(temp_video_path)
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_files, temp_dir)
        
        # Return results based on format
        if return_format == "json":
            # Convert events to serializable format
            serializable_events = []
            for event in results['events']:
                serializable_events.append({
                    'timestamp': event.timestamp,
                    'timestamp_formatted': format_timestamp(event.timestamp),
                    'event_type': event.event_type,
                    'confidence': event.confidence,
                    'duration': event.duration,
                    'details': event.details
                })
            
            return {
                'events': serializable_events,
                'report': results['report'],
                'metadata': results['metadata'],
                'processing_info': {
                    'filename': video_file.filename,
                    'config_preset': config_preset,
                    'analysis_timestamp': datetime.now().isoformat()
                }
            }
        
        elif return_format == "csv":
            # Export to CSV
            if results['events']:
                data = []
                for event in results['events']:
                    row = {
                        'timestamp': event.timestamp,
                        'timestamp_formatted': format_timestamp(event.timestamp),
                        'event_type': event.event_type,
                        'confidence': event.confidence,
                        'duration': event.duration or 0
                    }
                    
                    # Add details as separate columns
                    for key, value in event.details.items():
                        if isinstance(value, (str, int, float)):
                            row[f'detail_{key}'] = value
                    
                    data.append(row)
                
                df = pd.DataFrame(data)
                df.to_csv(temp_csv_path, index=False)
                
                return FileResponse(
                    temp_csv_path,
                    media_type='text/csv',
                    filename=f"analysis_{video_file.filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                )
            else:
                raise HTTPException(status_code=200, detail="No events detected - empty CSV")
        
        else:  # return_format == "log" (default)
            # Export to log format
            log_content = export_results_to_log(results)
            
            # Write to temporary file
            with open(temp_log_path, 'w') as f:
                f.write(log_content)
            
            return FileResponse(
                temp_log_path,
                media_type='text/plain',
                filename=f"analysis_{video_file.filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )
    
    except Exception as e:
        # Clean up on error
        cleanup_temp_files(temp_dir)
        
        logging.error(f"Analysis failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Video analysis failed: {str(e)}"
        )

def cleanup_temp_files(temp_dir: str):
    """Clean up temporary files"""
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        logging.warning(f"Failed to clean up temporary files: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('video_monitoring_api.log')
    ]
)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0", 
        port=10000, 
       # Use single worker to avoid MediaPipe conflicts
    )

