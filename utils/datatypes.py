from typing import List, Callable, Tuple, Dict, Optional
from algorithms.color_feature_extractor import extract_hsv_histogram_features
from algorithms.texture_feature_extractor import extract_lbp_features
import numpy as np
import os
import tqdm
import base64
import io
import json
import matplotlib.pyplot as plt

class ImageItem:
    def __init__(self, path: str, features: np.ndarray):
        self.path = path
        self.features = features

class ImageVectorDatabase:
    def __init__(self, similarity_metric: Callable = None, 
                 edge_detection_kwargs: Dict = None, 
                 hsv_feature_extractor_kwargs: Dict = None, 
                 lbp_feature_extractor_kwargs: Dict = None):
        self.data_store = []
        # default to canny edge detector
        self.edge_detection_kwargs = edge_detection_kwargs or {'edge_detection_strategy':'canny', 'low_threshold': 0, 'high_threshold': 50}
        self.hsv_feature_extractor_kwargs = hsv_feature_extractor_kwargs
        self.lbp_feature_extractor_kwargs = lbp_feature_extractor_kwargs
        
    
    def extract_features(self, image_path: str) -> np.ndarray:
        """
        Extract and combine HSV histogram and LBP features from an image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            np.ndarray: Concatenated feature vector
        """
        # Extract HSV color histogram features
        hsv_features = extract_hsv_histogram_features(
            image_path=image_path,
            **self.hsv_feature_extractor_kwargs,
            **self.edge_detection_kwargs
        )
        
        # Extract LBP texture features
        lbp_features = extract_lbp_features(
            image_path=image_path,
            **self.lbp_feature_extractor_kwargs,
            **self.edge_detection_kwargs
        )
        
        # Concatenate the features
        return np.concatenate([hsv_features, lbp_features])
    
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
        
        self.data_store = []  # Reset the database
        
        # Get list of image files first
        image_files = [os.path.join(folder_path, filename) 
                    for filename in os.listdir(folder_path) 
                    if filename.lower().endswith(extensions)]
        
        print(f"Building database from {folder_path} with {len(image_files)} images")
        
        # Use tqdm for progress tracking
        for image_path in tqdm.tqdm(image_files, desc="Processing images"):
            try:
                features = self.extract_features(image_path)
                self.data_store.append(ImageItem(image_path, features))
            except Exception as e:
                print(f"\nError processing {image_path}: {e}")
        
        print(f"Database built with {len(self.data_store)} images")
    
    def search(self, query_image_path: str, similarity_metric: Callable = None, k: int = 5, include_features: List[str] = None, feature_bins: Dict[str, int] = None) -> List[Tuple[str, float]]:
        """
        Search for the k most similar images to the query image.
        
        Args:
            query_image_path (str): Path to the query image
            k (int): Number of similar images to return
            
        Returns:
            List[Tuple[str, float]]: List of (image_path, similarity_score) pairs
        """
        
        # extract features from the query image
        query_features = self.extract_features(query_image_path)
        
        # compute similarity scores for all images in the database
        similarities = []
        for item in self.data_store:
            # ignore the query image
            if os.path.abspath(item.path) == os.path.abspath(query_image_path):
                continue
            if include_features == 'color':
                query_features = query_features[:feature_bins['color']]
                item_features = item.features[:feature_bins['color']]
                score = similarity_metric(query_features, item_features)
                
            elif include_features == 'texture':
                query_features = query_features[feature_bins['color']:feature_bins['color']+feature_bins['texture']]
                item_features = item.features[feature_bins['color']:feature_bins['color']+feature_bins['texture']]  
                score = similarity_metric(query_features, item_features)
                
            elif include_features == 'both':
                query_features = query_features[:feature_bins['color']+feature_bins['texture']]
                item_features = item.features[:feature_bins['color']+feature_bins['texture']]
                score = similarity_metric(query_features, item_features)
            similarities.append((item.path, score))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

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
            'items': []
        }
        
        # convert each item
        for item in self.data_store:
            features_bytes = io.BytesIO()
            np.save(features_bytes, item.features, allow_pickle=False)
            features_encoded = base64.b64encode(features_bytes.getvalue()).decode('utf-8')
            
            data['items'].append({
                'path': item.path,
                'features': features_encoded
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
        db = cls(edge_detection_kwargs=data.get('edge_detection_kwargs', None),
                 hsv_feature_extractor_kwargs=data.get('hsv_feature_extractor_kwargs', None),
                 lbp_feature_extractor_kwargs=data.get('lbp_feature_extractor_kwargs', None)
                 )
        
        for item_data in data['items']:
            features = np.array(item_data['features'])
            
            db.data_store.append(ImageItem(item_data['path'], features))
        
        print(f"Database loaded from {json_path} with {len(db.data_store)} images")
        return db
        
    def visualize_results(self, query_image_path: str, **search_kwargs) -> None:
        """
        Visualize the query image and its k most similar images.
        
        Args:
            query_image_path (str): Path to the query image
            k (int): Number of similar images to display
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