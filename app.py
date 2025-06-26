import gradio as gr
import numpy as np
import os
from PIL import Image
import tempfile
from typing import List, Tuple, Dict, Callable, Optional
import json

from utils.datatypes import ImageItem, ImageVectorDatabase
from utils.similarity_metrics import (
    manhattan_distance, euclidean_distance, cosine_similarity,
    shape_similarity_metric
)

def load_database(db_path: str):
    """Load the database from a pickle or JSON file"""
    try:
        if db_path.endswith('.pkl'):
            return ImageVectorDatabase.load_from_pickle(db_path)
        else:
            return ImageVectorDatabase.load_from_json(db_path)
    except Exception as e:
        print(f"Error loading database: {e}")
        return None

def search_similar_images(
    input_image,
    db: ImageVectorDatabase,
    feature_types: List[str],
    color_texture_similarity_metric: str,
    k: int,
    feature_weights: Dict[str, float] = None
) -> List[Tuple[str, Image.Image, float]]:
    """
    Search for similar images using the selected features and metrics.
    
    Args:
        input_image: The uploaded image (PIL Image)
        db: The image vector database
        feature_types: List of feature types to include ("color", "texture", "shape")
        color_texture_similarity_metric: "l1", "l2", or "cosine" for color/texture features
        k: Number of results to return
        feature_weights: Dictionary mapping feature types to weights
        
    Returns:
        List of (image_path, image, score) tuples
    """
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        temp_path = temp_file.name
        input_image.save(temp_path)
    
    if color_texture_similarity_metric == "l1":
        color_texture_metric = manhattan_distance
    elif color_texture_similarity_metric == "l2":
        color_texture_metric = euclidean_distance
    elif color_texture_similarity_metric == "cosine":
        color_texture_metric = cosine_similarity
    else:
        color_texture_metric = euclidean_distance  # Default
    
    shape_metric = shape_similarity_metric
    
    results = db.search(
        query_image_path=temp_path,
        k=k,
        feature_types=feature_types,
        feature_weights=feature_weights,
        color_similarity_metric=color_texture_metric,
        texture_similarity_metric=color_texture_metric,
        shape_similarity_metric=shape_metric
    )
    
    # Clean up the temp file
    os.unlink(temp_path)
    
    # Load the image data for each result
    result_items = []
    for img_path, score in results:
        try:
            img = Image.open(img_path)
            result_items.append((img_path, img, float(score)))
        except Exception as e:
            print(f"Error loading result image {img_path}: {e}")
    
    return result_items

def format_results(results: List[Tuple[str, Image.Image, float]]) -> List[Tuple[Image.Image, str]]:
    """Format the results for the Gradio gallery"""
    return [(img, f"Score: {score:.4f}\n{os.path.basename(path)}") for path, img, score in results]

def create_similarity_search_interface(db: ImageVectorDatabase):
    """Create and launch the Gradio interface"""

    # Define the interface
    with gr.Blocks(title="Image Similarity Search") as demo:
        gr.Markdown("# Image Similarity Search")
        gr.Markdown("Upload an image to find similar images in the database using color, texture, and shape features.")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input components
                input_image = gr.Image(type="pil", label="Input Image")
                
                # Feature type selection (multiple selection)
                feature_checkboxes = gr.CheckboxGroup(
                    choices=["color_features", "texture_features", "shape_features"],
                    value=["color_features", "texture_features"],
                    label="Feature Types to Include"
                )
                
                # Similarity metric for color/texture features only
                similarity_metric = gr.Dropdown(
                    choices=["l1", "l2", "cosine"],
                    value="l2",
                    label="Color/Texture Similarity Metric",
                    info="Shape features use a specialized similarity metric"
                )
                
                k = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    label="Number of Results"
                )
                
                search_button = gr.Button("Search", variant="primary")
            
            with gr.Column(scale=2):
                gallery = gr.Gallery(
                    label="Similar Images",
                    show_label=True,
                    elem_id="gallery",
                    columns=5,
                    object_fit="contain",
                    height="auto"
                )
        
        # Define the search function
        def search_wrapper(image, features, metric, k_val):
            if image is None:
                return []
            
            if not features:
                return []
            
            # Use equal weights for all selected features
            feature_weights = {feature.replace('_features', ''): 1.0 for feature in features}
            
            try:
                results = search_similar_images(
                    input_image=image,
                    db=db,
                    feature_types=features,
                    color_texture_similarity_metric=metric,
                    k=k_val,
                    feature_weights=feature_weights
                )
                
                return format_results(results)
                
            except Exception as e:
                return []
        
        search_button.click(
            fn=search_wrapper,
            inputs=[input_image, feature_checkboxes, similarity_metric, k],
            outputs=[gallery]
        )
        
        # Add some example usage information
        with gr.Row():
            gr.Markdown("""
            ### Usage Tips:
            - **Color features**: Good for finding images with similar color distributions
            - **Texture features**: Good for finding images with similar texture patterns (fur, skin, etc.)
            - **Shape features**: Good for finding images with similar object shapes and boundaries
            - **All selected features use equal weights**
            - **Similarity metrics**: 
              - L1 (Manhattan): Sum of absolute differences
              - L2 (Euclidean): Standard distance measure
              - Cosine: Angle-based similarity (good for normalized features)
            - Shape features automatically use a specialized similarity metric optimized for shape descriptors
            """)
    
    return demo

# Launch the interface
if __name__ == "__main__":
    db_base_path = r"C:\Users\VICTUS\Desktop\BTL\Hệ_cơ_sở_dữ_liệu_đa_phương_tiện\new_vectordb.json"

    if os.path.exists(db_base_path):
        print(f"Loading database from {db_base_path}")
        db = load_database(db_base_path)
    else:
        db = None
    
    if db is None:
        print("No database found! Please create a database first.")
        exit(1)  # Exit the program if no database is available
    
    # Create and launch the interface
    demo = create_similarity_search_interface(db)
    demo.launch(share=True)