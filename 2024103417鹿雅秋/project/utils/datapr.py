import numpy as np
import pandas as pd
import os
from typing import Tuple, List, Dict
from pathlib import Path
from tqdm import tqdm

class HighDDataProcessor:
    def __init__(self, data_path: str, processed_data_path: str = None, 
                 chunk_size: int = 5000, sequence_length: int = 30, 
                 prediction_horizon: int = 5):
        self.data_path = data_path
        # 如果未提供processed_data_path，则使用默认路径
        if processed_data_path is None:
            self.processed_data_path = os.path.join(os.path.dirname(data_path), "processed_data")
        else:
            self.processed_data_path = processed_data_path
        self.chunk_size = chunk_size
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        # 创建处理数据的目录
        os.makedirs(self.processed_data_path, exist_ok=True)
        
    def process_all_recordings(self, save=True) -> Dict:
        """处理所有记录并保存"""
        processed_data = {}
        
        # Get all trajectory files
        track_files = [f for f in os.listdir(self.data_path) 
                      if f.endswith('_tracks.csv')]
        
        for track_file in track_files:
            recording_id = int(track_file.split('_')[0])
            print(f"Processing recording {recording_id}...")
            
            try:
                # Process data in chunks
                self.process_single_recording(recording_id, save)
                print(f"Successfully processed recording {recording_id}")
            except Exception as e:
                print(f"Error processing recording {recording_id}: {str(e)}")
                continue
        
        return processed_data
    
    def process_single_recording(self, recording_id: int, save=True):
        """分块处理单个记录"""
        print_section(f"Processing Recording {recording_id}")
        
        tracks_path = os.path.join(self.data_path, f"{recording_id:02d}_tracks.csv")
        total_rows = sum(1 for _ in open(tracks_path)) - 1  # Subtract header row
        
        print_step("Data Info", 
                   f"Total rows: {total_rows}\n"
                   f"Chunk size: {self.chunk_size}\n"
                   f"Number of chunks: {(total_rows + self.chunk_size - 1) // self.chunk_size}")
        
        # Prepare storage for all processed data
        all_sequence_X = []
        all_sequence_y = []
        all_raw_features = []  # Store raw features for computing statistics
        
        # Process data in chunks
        for chunk_idx in tqdm(range(num_chunks), desc=f"Processing chunks"):
            # Read data chunk
            skip_rows = chunk_idx * self.chunk_size + 1 if chunk_idx > 0 else 0
            nrows = min(self.chunk_size, total_rows - chunk_idx * self.chunk_size)
            
            try:
                tracks_chunk = pd.read_csv(
                    tracks_path, 
                    skiprows=skip_rows, 
                    nrows=nrows,
                    usecols=['x', 'y', 'xVelocity', 'yVelocity'],
                    dtype={
                        'x': np.float32,
                        'y': np.float32,
                        'xVelocity': np.float32,
                        'yVelocity': np.float32
                    }
                )
                
                # Data validation
                if tracks_chunk.isnull().any().any():
                    print(f"Warning: Chunk {chunk_idx} contains null values, will interpolate")
                    tracks_chunk = tracks_chunk.interpolate(method='linear')
                
                # Extract features and perform simple scaling instead of standardization
                features = self.extract_features_from_chunk(tracks_chunk)
                
                # Calculate velocities and acceleration
                velocities = np.sqrt(features[:, 2]**2 + features[:, 3]**2)  # Velocity magnitude
                mean_velocity = np.mean(velocities)
                
                if chunk_idx == 0:
                    print_step("Raw Data Statistics", 
                              f"Position range x: [{features[:, 0].min():.3f}, {features[:, 0].max():.3f}]\n"
                              f"Position range y: [{features[:, 1].min():.3f}, {features[:, 1].max():.3f}]\n"
                              f"Velocity range x: [{features[:, 2].min():.3f}, {features[:, 2].max():.3f}]\n"
                              f"Velocity range y: [{features[:, 3].min():.3f}, {features[:, 3].max():.3f}]\n"
                              f"Mean velocity: {mean_velocity:.3f}")
                
                all_raw_features.append(features)
                
            except Exception as e:
                print(f"Error processing chunk {chunk_idx}: {str(e)}")
                continue
        
        try:
            # Combine all raw features
            all_features = np.concatenate(all_raw_features, axis=0)
            
            # Calculate feature ranges
            feature_min = np.min(all_features, axis=0)
            feature_max = np.max(all_features, axis=0)
            feature_range = feature_max - feature_min
            
            # Avoid division by zero
            feature_range = np.where(feature_range < 1e-6, 1.0, feature_range)
            
            # Scale all features to [-1, 1] range using min-max scaling
            for features in all_raw_features:
                # Use min-max scaling instead of standardization
                normalized_features = 2 * (features - feature_min) / feature_range - 1
                
                # Prepare sequence data
                X, y = self.prepare_sequence_data(
                    normalized_features, 
                    self.sequence_length, 
                    self.prediction_horizon
                )
                
                if len(X) > 0:
                    all_sequence_X.append(X)
                    all_sequence_y.append(y)
            
            # Combine all chunks' data
            sequence_X = np.concatenate(all_sequence_X, axis=0) if all_sequence_X else np.array([], dtype=np.float32)
            sequence_y = np.concatenate(all_sequence_y, axis=0) if all_sequence_y else np.array([], dtype=np.float32)
            
            # Validate processed data
            print(f"\nProcessed data statistics:")
            print(f"Input sequence shape: {sequence_X.shape}")
            print(f"Target sequence shape: {sequence_y.shape}")
            print(f"Input sequence range: [{sequence_X.min():.3f}, {sequence_X.max():.3f}]")
            print(f"Target sequence range: [{sequence_y.min():.3f}, {sequence_y.max():.3f}]")
            
            # Save processed data and scaling parameters
            if save and len(sequence_X) > 0:
                self._save_processed_data(recording_id, {
                    'sequence_X': sequence_X,
                    'sequence_y': sequence_y,
                    'stats': {
                        'min': feature_min,
                        'max': feature_max,
                        'range': feature_range
                    }
                })
                
            # Clean up memory
            del all_raw_features, all_sequence_X, all_sequence_y
            
        except Exception as e:
            print(f"Error merging data: {str(e)}")
            raise
    
    def extract_features_from_chunk(self, tracks_chunk: pd.DataFrame) -> np.ndarray:
        """从数据块中提取特征"""
        return tracks_chunk[['x', 'y', 'xVelocity', 'yVelocity']].values.astype(np.float32)
    
    def prepare_sequence_data(self, 
                            trajectory_features: np.ndarray,
                            sequence_length: int,
                            prediction_horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """准备用于训练的序列数据"""
        if len(trajectory_features) <= sequence_length + prediction_horizon:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        
        # Use sliding window to create sequences
        total_length = len(trajectory_features)
        num_sequences = total_length - sequence_length - prediction_horizon + 1
        
        # Pre-allocate memory
        X = np.zeros((num_sequences, sequence_length, trajectory_features.shape[1]), dtype=np.float32)
        y = np.zeros((num_sequences, prediction_horizon, trajectory_features.shape[1]), dtype=np.float32)
        
        # Fill data
        for i in range(num_sequences):
            X[i] = trajectory_features[i:i+sequence_length]
            y[i] = trajectory_features[i+sequence_length:i+sequence_length+prediction_horizon]
        
        return X, y
    
    def load_processed_data(self, recording_id: int) -> Dict:
        """加载处理后的数据"""
        processed_file = os.path.join(self.processed_data_path, f"recording_{recording_id:02d}.npz")
        if not os.path.exists(processed_file):
            raise FileNotFoundError(f"Processed data file {processed_file} does not exist")
            
        try:
            data = np.load(processed_file, allow_pickle=True)
            return {
                'sequence_X': data['sequence_X'],
                'sequence_y': data['sequence_y'],
                'stats': data['stats'].item()
            }
        except Exception as e:
            print(f"Error loading data file {processed_file}: {str(e)}")
            raise
    
    def _save_processed_data(self, recording_id: int, data: Dict):
        """保存处理后的数据"""
        save_path = os.path.join(self.processed_data_path, f"recording_{recording_id:02d}.npz")
        try:
            np.savez(
                save_path,
                sequence_X=data['sequence_X'],
                sequence_y=data['sequence_y'],
                stats=data['stats']
            )
            print(f"Saved processed data to {save_path}")
        except Exception as e:
            print(f"Error saving data to {save_path}: {str(e)}")
            raise
    
    def test_process_data(self, recording_id: int):
        """测试数据处理结果"""
        print(f"\nTesting data processing results for recording {recording_id}...")
        
        # Read a small portion of original data for testing
        tracks_path = os.path.join(self.data_path, f"{recording_id:02d}_tracks.csv")
        test_chunk = pd.read_csv(
            tracks_path,
            nrows=100,  # Read only the first 100 rows for testing
            usecols=['x', 'y', 'xVelocity', 'yVelocity']
        )
        
        print("\nOriginal data sample:")
        print(test_chunk.head())
        print("\nOriginal data statistics:")
        print(test_chunk.describe())
        
        # Process test data
        features = self.extract_features_from_chunk(test_chunk)
        
        # Calculate velocities
        velocities = np.sqrt(features[:, 2]**2 + features[:, 3]**2)
        
        print("\nCalculated velocity statistics:")
        print(f"Minimum velocity: {velocities.min():.3f} m/s")
        print(f"Maximum velocity: {velocities.max():.3f} m/s")
        print(f"Mean velocity: {velocities.mean():.3f} m/s")
        
        # Calculate feature ranges
        feature_min = np.min(features, axis=0)
        feature_max = np.max(features, axis=0)
        feature_range = feature_max - feature_min
        feature_range = np.where(feature_range < 1e-6, 1.0, feature_range)
        
        # Perform min-max scaling
        normalized_features = 2 * (features - feature_min) / feature_range - 1
        
        print("\nNormalized data statistics:")
        print(f"Position x range: [{normalized_features[:, 0].min():.3f}, {normalized_features[:, 0].max():.3f}]")
        print(f"Position y range: [{normalized_features[:, 1].min():.3f}, {normalized_features[:, 1].max():.3f}]")
        print(f"Velocity x range: [{normalized_features[:, 2].min():.3f}, {normalized_features[:, 2].max():.3f}]")
        print(f"Velocity y range: [{normalized_features[:, 3].min():.3f}, {normalized_features[:, 3].max():.3f}]")
        
        return normalized_features 