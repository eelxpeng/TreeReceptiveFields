import numpy as np
import os
from PIL import Image

mask = np.load("layer1-mask.npy")
n_obs, n_hidden = mask.shape
directory = "maskVisual"
if not os.path.exists(directory):
    os.makedirs(directory)
for i in range(n_hidden):
    m = mask[:, i]*255
    image = m.reshape([28,28]).astype("uint8")
    im = Image.fromarray(image)
    im.save(directory+"/h-"+str(i)+".jpg")
