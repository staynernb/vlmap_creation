from PIL import Image, ImageDraw
import requests
import cv2
from torchvision.utils import save_image
import numpy as np
from torchvision import transforms
import sys

#url = "https://github.com/timojl/clipseg/blob/master/example_image.jpg?raw=true"
#image = Image.open(requests.get(url, stream=True).raw)
#new_image = image.resize((352, 352))
#new_image.save("resized_img.jpeg")

image = Image.open("./image/camera_image.jpeg")

## Crop image to avoid distortion
#desired_width = 480
#desired_height = 480
#(width, height) = image.size
#left = int((width - desired_width)/2)
#right = left + desired_width
#top = int((height - desired_height)/2)
#bottom = top + desired_height
#new_img = image.crop((left, top, right, bottom))
#new_img = new_img.resize((desired_width,desired_height))

#new_img.save("/app/image/image_resized.jpeg")
#new_img.show()
#image.show()

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

prompts = [sys.argv[1]]

inputs = processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt")

import torch
import matplotlib.pyplot as plt

# predict
with torch.no_grad():
  outputs = model(**inputs)



img1 = torch.sigmoid(outputs.logits)
min = img1.min()
max = img1.max()
#print(max)
img2 = 1./(max-min) * img1 + 1.*min / (min-max)

save_image(img2, '/app/image/image_result.jpeg')

image = Image.open("/app/image/image_result.jpeg").convert('L')
img_resu = image.resize((620,480))
img_resu.save("/app/image/image_result_resized.jpeg")
#img_resu = img_resu.convert('1')

#img1 = torch.sigmoid(outputs.logits)
#min = img_resu.min()
#max = img_resu.max()
#print(max)
#img2 = 1./(max-min) * img_resu + 1.*min / (min-max)

#save_image(img2, 'GREY_img.png')
#save_image(img1, 'RGB_img.png')
img_resu_np = np.array(img_resu)

# Find indices where we have mass
mass_y, mass_x = np.where(img_resu_np >= 200)
# mass_x and mass_y are the list of x indices and y indices of mass pixels
cent_x = np.average(mass_x)
cent_y = np.average(mass_y)

theta = float((np.arctan2((cent_x-320.5),530.4669406576809)*180/np.pi))
print("theta: ", round(theta))
print("cent_x: ", cent_x)

draw = ImageDraw.Draw(img_resu)
draw.ellipse([(cent_x-10,cent_y-10),(cent_x+10,cent_y+10)], fill="white", outline="black", width=5)
img_resu.save("/app/image/image_result_resized_withcenter.jpeg")

sys.exit(-round(theta))
  


