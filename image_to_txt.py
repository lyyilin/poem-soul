
# from transformers import BlipProcessor, BlipForConditionalGeneration
# class image_to_txt():
#   def __init__(self):
#       self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
#       self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

#   def to_txt(self,image):
#       i_image = image
#       if i_image.mode != "RGB":
#           i_image = i_image.convert(mode="RGB")

#       inputs = self.processor(i_image, return_tensors="pt")

#       out = self.model.generate(**inputs)
#       preds = self.processor.decode(out[0], skip_special_tokens=True)


#       return preds


# if __name__ == '__main__':
#     my_class = image_to_txt()
#     text = my_class.to_txt(r'E:\my_item\poem_picture\model1.png')
#     print(text[0])


from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

class image_to_txt():

  def __init__(self):
    # self.model_path = r"./models"
    self.model = VisionEncoderDecoderModel.from_pretrained(r"./models")
    self.feature_extractor = ViTImageProcessor.from_pretrained(r"./models")
    self.tokenizer = AutoTokenizer.from_pretrained(r"./models")


  def to_txt(self,image):
    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model.to(device)

    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    # i_image = Image.open(image_path)
    i_image = image
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    pixel_values = self.feature_extractor(images=i_image,return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = self.model.generate(pixel_values, **gen_kwargs)

    preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds


if __name__ == '__main__':
    my_class = image_to_txt()
    text = my_class.to_txt(r'E:\my_item\poem_picture\model1.png')
    print(text[0])
