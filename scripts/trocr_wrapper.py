from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import io

class TrOCROCR:
    def __init__(self, model_name="microsoft/trocr-base-handwritten"):
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)

    def ocr_bytes(self, img_bytes):
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(inputs)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
