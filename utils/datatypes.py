from typing import List, Callable, Tuple, Dict, Optional
# from algorithms.color_feature_extractor import extract_hsv_histogram_features
# from algorithms.texture_feature_extractor import extract_lbp_features
from algorithms.color_feature_extractor import extract_hsv_histogram_features
from algorithms.texture_feature_extractor import extract_lbp_features
from algorithms.ltp_texture_feature_extractor import extract_ltp_features
from algorithms.shape_feature_extractor import extract_shape_features
import numpy as np
import os
import tqdm
import base64
import io
import json
import matplotlib.pyplot as plt
import pickle

class ImageItem:
    def __init__(self, path: str, features: Dict[str, any]):
        self.path = path
        self.features = features  # Now a dictionary with keys: 'color_features', 'texture_features', 'shape_features'

class ImageVectorDatabase:
    def __init__(self, 
                 edge_detection_kwargs: Dict = None, 
                 hsv_feature_extractor_kwargs: Dict = None, 
                 lbp_feature_extractor_kwargs: Dict = None,
                 shape_feature_extractor_kwargs: Dict = None,
                 color_similarity_metric: Callable = None,
                 texture_similarity_metric: Callable = None,
                 shape_similarity_metric: Callable = None):
        self.data_store = []
        # default to canny edge detector
        self.edge_detection_kwargs = edge_detection_kwargs or {'edge_detection_strategy':'canny', 'low_threshold': 0, 'high_threshold': 50}
        self.hsv_feature_extractor_kwargs = hsv_feature_extractor_kwargs or {}
        self.lbp_feature_extractor_kwargs = lbp_feature_extractor_kwargs or {}
        self.shape_feature_extractor_kwargs = shape_feature_extractor_kwargs or {}
        self.color_similarity_metric = color_similarity_metric
        self.texture_similarity_metric = texture_similarity_metric
        self.shape_similarity_metric = shape_similarity_metric
        
    
    def extract_features(self, image_path: str) -> Dict[str, np.ndarray]:
        """
        Extract color, texture, and shape features from an image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Dict[str, np.ndarray]: Dictionary containing different feature types
        """
        features = {}
        
        # Extract HSV color histogram features
        features['color_features'] = extract_hsv_histogram_features(
            image_path=image_path,
            **self.hsv_feature_extractor_kwargs,
            **self.edge_detection_kwargs
        )
        
        # Extract LBP texture features
        features['texture_features'] = extract_ltp_features(
            image_path=image_path,
            **self.lbp_feature_extractor_kwargs,
            **self.edge_detection_kwargs
        )
       
        shape_features_dict = extract_shape_features(image_path)
        features['shape_features'] = shape_features_dict
        
        return features
    
    def add_image(self, image_path: str) -> None:
        """
        Add a single image to the database.
        
        Args:
            image_path (str): Path to the image file
        """
        try:
            features = self.extract_features(image_path)
            self.data_store.append(ImageItem(image_path, features))
            print(f"Added {image_path} to database")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    def build_from_folder(self, folder_path: str, extensions: Tuple[str] = ('.jpg', '.jpeg', '.png', '.bmp')) -> None:
        """
        Build the database by processing all images in a folder.
        
        Args:
            folder_path (str): Path to the folder containing images
            extensions (tuple): Valid image file extensions to process
        """
        if not os.path.isdir(folder_path):
            raise ValueError(f"Folder not found: {folder_path}")
        
        self.data_store = [] 
        
        # Get list of image files first
        image_files = [os.path.join(folder_path, filename) 
                    for filename in os.listdir(folder_path) 
                    if filename.lower().endswith(extensions)]
        
        print(f"Building database from {folder_path} with {len(image_files)} images")
        
        for image_path in tqdm.tqdm(image_files, desc="Processing images"):
            try:
                features = self.extract_features(image_path)
                self.data_store.append(ImageItem(image_path, features))
            except Exception as e:
                print(f"\nError processing {image_path}: {e}")
        
        print(f"Database built with {len(self.data_store)} images")
    
    def _compute_similarity_score(self, query_features: Dict, item_features: Dict, 
                                feature_types: List[str], feature_weights: Dict[str, float] = None) -> float:
        """
        Compute similarity score between query and item features.
        
        Args:
            query_features: Query image features
            item_features: Database item features  
            feature_types: List of feature types to include ('color', 'texture', 'shape')
            feature_weights: Weights for different feature types
            
        Returns:
            Combined similarity score
        """
        if feature_weights is None:
            feature_weights = {'color': 1.0, 'texture': 1.0, 'shape': 1.0}
        
        total_score = 0.0
        total_weight = 0.0
        
        for feature_type in feature_types:
            weight = feature_weights.get(feature_type, 1.0)
            
            if feature_type == 'color_features':
                if self.color_similarity_metric is not None:
                    score = self.color_similarity_metric(
                        query_features['color_features'], 
                        item_features['color_features']
                    )
                    
            elif feature_type == 'texture_features':
                if self.texture_similarity_metric is not None:
                    score = self.texture_similarity_metric(
                        query_features['texture_features'], 
                        item_features['texture_features']
                    )
                
            elif feature_type == 'shape_features':
                if self.shape_similarity_metric is not None:
                    score = self.shape_similarity_metric(
                        query_features['shape_features'], 
                        item_features['shape_features']
                    )
            else:
                raise ValueError(f"Invalid feature type: {feature_type}")
                
            total_score += weight * score
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0

    def search(self, query_image_path: str, k: int = 5, 
              feature_types: List[str] = ['color', 'texture', 'shape'], 
              feature_weights: Dict[str, float] = None,
              color_similarity_metric: Callable = None,
              texture_similarity_metric: Callable = None,
              shape_similarity_metric: Callable = None) -> List[Tuple[str, float]]:
        """
        Search for the k most similar images to the query image.
        
        Args:
            query_image_path (str): Path to the query image
            k (int): Number of similar images to return
            feature_types (List[str]): Feature types to include ('color', 'texture', 'shape')
            feature_weights (Dict[str, float]): Weights for different feature types
            color_similarity_metric: Similarity metric for color features
            texture_similarity_metric: Similarity metric for texture features
            shape_similarity_metric: Similarity metric for shape features
            
        Returns:
            List[Tuple[str, float]]: List of (image_path, similarity_score) pairs
        """
        
        if color_similarity_metric is not None:
            self.color_similarity_metric = color_similarity_metric
        if texture_similarity_metric is not None:
            self.texture_similarity_metric = texture_similarity_metric
        if shape_similarity_metric is not None:
            self.shape_similarity_metric = shape_similarity_metric
        
        # extract features from the query image
        query_features = self.extract_features(query_image_path)
        
        # compute similarity scores for all images in the database
        similarities = []
        for item in self.data_store:
            # ignore the query image
            if os.path.abspath(item.path) == os.path.abspath(query_image_path):
                continue
                
            score = self._compute_similarity_score(
                query_features, item.features, feature_types, feature_weights
            )
            similarities.append((item.path, score))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def _numpy_to_list(self, obj):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, list):
            return [self._numpy_to_list(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: self._numpy_to_list(v) for k, v in obj.items()}
        else:
            return obj

    def save_to_json(self, json_path: str) -> None:
        """
        Save the database to a JSON file.
        
        Args:
            json_path (str): Path to save the JSON file
        """
        data = {
            'edge_detection_kwargs': self.edge_detection_kwargs,
            'hsv_feature_extractor_kwargs': self.hsv_feature_extractor_kwargs,
            'lbp_feature_extractor_kwargs': self.lbp_feature_extractor_kwargs,
            'shape_feature_extractor_kwargs': self.shape_feature_extractor_kwargs,
            'items': []
        }
        
        # convert each item
        for item in self.data_store:
            # Convert features to JSON-serializable format
            features_serialized = {}
            for feature_type, feature_data in item.features.items():
                if isinstance(feature_data, np.ndarray):
                    features_serialized[feature_type] = {
                        'type': 'numpy_array',
                        'data': feature_data.tolist(),
                        'shape': feature_data.shape,
                        'dtype': str(feature_data.dtype)
                    }
                elif isinstance(feature_data, dict):
                    features_serialized[feature_type] = {
                        'type': 'dict',
                        'data': self._numpy_to_list(feature_data)
                    }
                else:
                    features_serialized[feature_type] = {
                        'type': 'other',
                        'data': feature_data
                    }
            
            data['items'].append({
                'path': item.path,
                'features': features_serialized
            })
        
        # save to file
        with open(json_path, 'w') as f:
            json.dump(data, f)
        
        print(f"Database saved to {json_path} with {len(self.data_store)} images")

    @classmethod
    def load_from_json(cls, json_path: str) -> 'ImageVectorDatabase':
        """
        Load a database from a JSON file.
        
        Args:
            json_path (str): Path to the JSON file
            
        Returns:
            ImageVectorDatabase: Loaded database
        """
        
        # load from file
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # create a new database
        db = cls(
            edge_detection_kwargs=data.get('edge_detection_kwargs', None),
            hsv_feature_extractor_kwargs=data.get('hsv_feature_extractor_kwargs', None),
            lbp_feature_extractor_kwargs=data.get('lbp_feature_extractor_kwargs', None),
            shape_feature_extractor_kwargs=data.get('shape_feature_extractor_kwargs', None)
        )
        
        for item_data in data['items']:
            # Deserialize features
            features = {}
            for feature_type, feature_info in item_data['features'].items():
                if feature_info['type'] == 'numpy_array':
                    # Reconstruct numpy array
                    features[feature_type] = np.array(feature_info['data'], dtype=feature_info['dtype'])
                elif feature_info['type'] == 'dict':
                    features[feature_type] = feature_info['data']
                else:
                    features[feature_type] = feature_info['data']
            
            db.data_store.append(ImageItem(item_data['path'], features))
        
        print(f"Database loaded from {json_path} with {len(db.data_store)} images")
        return db
        
    def visualize_results(self, query_image_path: str, **search_kwargs) -> None:
        """
        Visualize the query image and its k most similar images.
        
        Args:
            query_image_path (str): Path to the query image
            **search_kwargs: Arguments passed to search method
        """
        
        # get similar images
        results = self.search(query_image_path, **search_kwargs)
        
        plt.figure(figsize=(15, 4))
        
        k = search_kwargs.get('k', 5)
        # display query image
        plt.subplot(1, k+1, 1)
        query_img = plt.imread(query_image_path)
        plt.imshow(query_img)
        plt.title("Query Image")
        plt.axis('off')
        
        # Display similar images
        for i, (img_path, score) in enumerate(results):
            plt.subplot(1, k+1, i+2)
            similar_img = plt.imread(img_path)
            plt.imshow(similar_img)
            plt.title(f"Similarity: {score:.4f}\n{os.path.basename(img_path)}", fontsize=8)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()