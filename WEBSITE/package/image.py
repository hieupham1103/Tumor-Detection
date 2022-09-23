import requests
import numpy as np
from io import BytesIO
from PIL import Image

def url_to_img(url, save_as=''):
  img = Image.open(BytesIO(requests.get(url).content))
  if save_as:
    img.save(save_as)
  return np.array(img)