### Technical Details
#### Feature Extraction
*Color features*
- **HSV Color Histograms**: Images are converted to HSV color space and histograms are computed.
- HSV is more perceptually meaningful than RGB for similarity search.
Bin allocation: 8 bins for Hue, 12 for Saturation, and 3 for Value, resulting in a 144-dimensional feature vector.

*Texture features*
- **(Uniform) Local Binary Patterns (LBP)**: Captures textural patterns by comparing each pixel with its neighbors.
- Uniform LBP patterns are used to reduce feature dimensionality while preserving discriminative power.
- 59-dimensional feature vector representing texture patterns.
*Object Extraction*
- Background removal using contour detection to focus feature extraction on the main object.
- Canny edge detection to identify object boundaries.

*Shape features (incoming)*

#### Similarity Metrics
- **L1 (Manhattan) Distance**: Sum of absolute differences between feature vectors.
- **L2 (Euclidean) Distance**: Square root of the sum of squared differences.
- **Cosine Similarity**: Measures the cosine of the angle between vectors, focusing on orientation rather than magnitude.

### Setup instructions

#### Preresquisites
- Python 3.12.4

1. Clone the repository
```
git clone https://github.com/1-13AM/BTL_HCSDLDPT
cd BTL_HCSDLDPT
```

2. Create a virtual environment (recommended):
```
python -m venv venv

# Windows
venv\Scripts\activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Prepare the image database (the vectordb.json is available in the repo so you probably don't have to run this):
- Place images in a folder (e.g., animal_datasets_no_bg)

- Run feature extraction to build the vector database:
```
from utils.datatypes import ImageVectorDatabase

# Create and build database
db = ImageVectorDatabase()
db.build_from_folder("path/to/image/folder")
db.save_to_json("vectordb.json")
```

### Usage
Run the application and open the provided URL
```
python app.py
```