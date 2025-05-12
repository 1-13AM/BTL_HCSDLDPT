import gradio as gr
import numpy as np
import os
from PIL import Image
import tempfile
from typing import List, Tuple, Dict, Callable, Optional
import json

# Import necessary modules
from utils.datatypes import ImageItem, ImageVectorDatabase
from utils.similarity_metrics import (manhattan_distance, euclidean_distance, 
                                    cosine_similarity)

FEATURE_BINS = {'color': 144, 'texture': 59}

def load_database(db_path: str):
    """Load the database from a JSON file"""
    try:
        return ImageVectorDatabase.load_from_json(db_path)
    except Exception as e:
        print(f"Error loading database: {e}")
        return None

def search_similar_images(
    input_image,
    db: ImageVectorDatabase,
    include_features: str,
    similarity_metric: str,
    k: int,
    feature_bins: Dict[str, int]
) -> List[Tuple[str, Image.Image, float]]:
    """
    Search for similar images using the selected features and metric.
    
    Args:
        input_image: The uploaded image (PIL Image)
        db: The image vector database
        include_features: "color", "texture", or "both"
        similarity_metric: "l1", "l2", or "cosine"
        k: Number of results to return
        feature_bins: Dictionary mapping feature types to bin counts
        
    Returns:
        List of (image_path, image, score) tuples
    """
    # Save the input image to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        temp_path = temp_file.name
        input_image.save(temp_path)
    
    # Select the appropriate similarity metric
    if similarity_metric == "l1":
        metric_func = manhattan_distance
    elif similarity_metric == "l2":
        metric_func = euclidean_distance
    elif similarity_metric == "cosine":
        metric_func = cosine_similarity
    else:
        metric_func = euclidean_distance  # Default
    
    # Convert feature type to list
    if include_features == "color":
        features = "color"
    elif include_features == "texture":
        features = "texture"
    else:
        features = "both"
    
    # Search for similar images
    results = db.search(
        query_image_path=temp_path,
        similarity_metric=metric_func,
        k=k,
        include_features=features,
        feature_bins=feature_bins
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
        gr.Markdown("Upload an image to find similar images in the database.")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input components
                input_image = gr.Image(type="pil", label="Input Image")
                
                include_features = gr.Radio(
                    choices=["color", "texture", "both"],
                    value="both",
                    label="Feature Type"
                )
                
                similarity_metric = gr.Dropdown(
                    choices=["l1", "l2", "cosine"],
                    value="l2",
                    label="Similarity Metric"
                )
                
                k = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    label="Number of Results"
                )
                
                search_button = gr.Button("Search")
            
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
            results = search_similar_images(
                input_image=image, 
                db=db, 
                include_features=features, 
                similarity_metric=metric, 
                k=k_val, 
                feature_bins=FEATURE_BINS
            )
            return format_results(results)
        
        search_button.click(
            fn=search_wrapper,
            inputs=[input_image, include_features, similarity_metric, k],
            outputs=gallery
        )
    
    return demo

# Launch the interface
if __name__ == "__main__":
    db_path = r"C:\Users\VICTUS\Desktop\BTL\Hệ_cơ_sở_dữ_liệu_đa_phương_tiện\vectordb.json"
    db = load_database(db_path)
    
    # from PIL import Image
    # image = Image.open(r"C:\Users\VICTUS\Desktop\BTL\Hệ_cơ_sở_dữ_liệu_đa_phương_tiện\animal_datasets_no_bg\gettyimages-73319220-612x612.png")
    # results = search_similar_images(input_image=image,
    #                                db=db,
    #                                include_features="both",
    #                                similarity_metric="l2",
    #                                k=5,
    #                                feature_bins=FEATURE_BINS)
    # print(results)
    
    # images = db.search(query_image_path=r"C:\Users\VICTUS\Desktop\BTL\Hệ_cơ_sở_dữ_liệu_đa_phương_tiện\animal_datasets_no_bg\gettyimages-73319220-612x612.png",
    #                    similarity_metric=euclidean_distance,
    #                    k=5,
    #                    include_features="both",
    #                    feature_bins=FEATURE_BINS)
    # print(images)
    
    demo = create_similarity_search_interface(db)
    demo.launch(share=True)