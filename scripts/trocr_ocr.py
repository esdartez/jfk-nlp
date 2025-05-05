from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import os
import torch

class TrOCROCR:
    def __init__(self, model_name="microsoft/trocr-base-handwritten"):
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)

    def ocr_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text

if __name__ == "__main__":
    ocr = TrOCROCR()
    test_img = "sample_page.png"
    result = ocr.ocr_image(test_img)
    print(result)
