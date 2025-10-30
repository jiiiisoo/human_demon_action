import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
from IPython import display

def as_gif(images, path="temp.gif"):
  # Render the images as the gif (15Hz control frequency):
  images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000/15), loop=0)
  gif_bytes = open(path,"rb").read()
  return gif_bytes


ds = tfds.load("droid_100", data_dir="gs://gresearch/robotics", split="train")

images = []
for episode in ds.shuffle(10, seed=0).take(1):
  for i, step in enumerate(episode["steps"]):
    images.append(
      Image.fromarray(
        np.concatenate((
              step["observation"]["exterior_image_1_left"].numpy(),
              step["observation"]["exterior_image_2_left"].numpy(),
              step["observation"]["wrist_image_left"].numpy(),
        ), axis=1)
      )
    )

display.Image(as_gif(images))