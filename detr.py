import os
import sys
from datetime import datetime
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import torch
import matplotlib.pyplot as plt
from transformers import DetrImageProcessor, DetrForObjectDetection, DetrForSegmentation
import numpy as np

def open_image(image_path, modifications=None):
    if modifications==None:
        return Image.open(image_path)

class DETR():
    """
    DETR class for handling image processing tasks.

    Arguments:
    image_path : str
        Path to the image file.
    save_path : str
        Path to save the results.
    image_name : str, optional
        Optional name of the image file (default is None).
    segmentation : bool, optional
        Flag to indicate if segmentation is required (default is False).
    save : bool, optional
        Flag to indicate if results should be saved (default is False).
    """
    def __init__(self,
                image_path,
                save_path,
                image_name=None,
                segmentation=False,
                save=False):
        self.image_path = image_path
        self.image_name = image_name # Unused
        self.image = open_image(self.image_path)  # Use PIL.Image.open directly
        self.segmentation = segmentation
        
        self.save = save
        self.save_path = save_path

        if self.segmentation:
            self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50-panoptic") # Or resnet101
            self.model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic") # Or resnet101
        else:
            self.processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-101-dc5')
            self.model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-101-dc5')
        
        self.inputs = self.processor(images=self.image, return_tensors="pt")
        pass

    def run(self):
        # Runs the model
        with torch.no_grad():
            self.outputs = self.model(**self.inputs)

    def show_results(self):
        # Show or save the results
        if self.segmentation:
            self.show_segmentation_results()
        else:
            self.show_detection_results()

    def show_detection_results(self):
        """
        Display the object detection results on the image.

        This method processes the detection results, draws bounding boxes and labels on the image,
        and either displays the image or saves it to the specified path.
        """
        target_sizes = torch.tensor([self.image.size[::-1]])
        results = self.processor.post_process_object_detection(self.outputs, target_sizes=target_sizes)[0]
        draw = ImageDraw.Draw(self.image)
        font = ImageFont.load_default()

        score_threshold = 0.5  # Adjust as needed
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if score > score_threshold:
                box = [round(i, 2) for i in box.tolist()]
                draw.rectangle(box, outline="red", width=3)
                label_text = f"{self.model.config.id2label[label.item()]}: {round(score.item()*100, 1)}"
                text_bbox = draw.textbbox((box[0], box[1]), label_text, font=font)
                draw.rectangle([text_bbox[0], text_bbox[1], text_bbox[2], text_bbox[3]], fill="red")
                draw.text((box[0], box[1]), label_text, fill="white", font=font)

        plt.figure(figsize=(10, 10))
        plt.imshow(self.image)
        plt.axis('off')
        if self.save:
            plt.savefig(os.path.join(self.save_path, 'object_detection.jpg'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def show_segmentation_results(self):
        """
        Display the segmentation results on the image.

        This method processes the segmentation results, applies a color map to the segmentation mask,
        overlays the mask on the original image, and either displays the image or saves it to the specified path.
        """
        panoptic_seg = self.processor.post_process_panoptic_segmentation(self.outputs, target_sizes=[self.image.size[::-1]])[0]
        panoptic_seg_img = panoptic_seg["segmentation"].cpu().numpy()
        segments_info = panoptic_seg["segments_info"]

        # Create a color map
        color_map = np.zeros((len(segments_info) + 1, 3), dtype=np.uint8)
        for i in range(1, len(segments_info) + 1):
            color_map[i] = np.random.randint(0, 256, 3)

        # Apply color map to segmentation mask
        colored_mask = color_map[panoptic_seg_img]

        # Overlay the mask on the original image
        plt.figure(figsize=(10, 10))
        plt.imshow(self.image)
        plt.imshow(colored_mask, alpha=0.5)
        plt.axis('off')
        if self.save:
            plt.savefig(os.path.join(self.save_path, 'segmentation.jpg'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()