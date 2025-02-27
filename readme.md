Sure, here is the updated README content with the images displayed at the end:

```markdown
# Classify Anything with DETR

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
    git clone https://github.com/yourusername/classify-anything.git
    cd classify-anything
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
```

### Segmentation

To run segmentation on an image, use the following code:

```python
from detr import DETR

# Initialize the DETR object
detr = DETR(
    image_path='path/to/your/image.jpg',
    save_path='path/to/save/results',
    segmentation=True,
    save=True
)

# Run the model
detr.run()

# Show or save the results
detr.show_results()
```

## Example

Here are examples of the object detection and segmentation results:

### Object Detection Results
![Object Detection Results](/results/object_detection.jpg)

### Segmentation Results
![Segmentation Results](/results/segmentation.jpg)

## Acknowledgements

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Facebook Research DETR](https://github.com/facebookresearch/detr)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

---

### Directory Structure

```
ClassifyAnything/
│
├── classification
    └── detr.py
├── README.md
└── results/
    ├── object_detection.jpg
    └── segmentation.jpg
