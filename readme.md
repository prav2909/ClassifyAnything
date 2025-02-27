# Classify Anything

This repository contains a Python implementation of the DETR (DEtection TRansformer) model from Hugging Face for object detection and segmentation tasks. The code uses the `transformers` library from Hugging Face and the `PIL` library for image processing. More models and functionality will be added soon.

## Requirements

- Python 3.6+
- torch
- transformers
- Pillow
- matplotlib
- numpy

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/prav2909/ClassifyAnything.git
    cd classifyAnything
    ```

2. Install the required packages:
    ```bash
    pip install torch transformers Pillow matplotlib numpy
    ```

## Usage

### Object Detection

To run object detection on an image, use the following code:

```python
from detr import DETR

# Initialize the DETR object
detr = DETR(
    image_path='path/to/your/image.jpg',
    save_path='path/to/save/results',
    segmentation=False,
    save=True
)

# Run the model
detr.run()

# Show or save the results
detr.show_results()
