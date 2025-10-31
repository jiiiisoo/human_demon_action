import numpy as np
import tensorflow_datasets as tfds
from PIL import Image
from IPython import display

def dataset2path(dataset_name):
  if dataset_name == 'robo_net':
    version = '1.0.0'
  elif dataset_name == 'language_table':
    version = '0.0.1'
  elif dataset_name == 'droid':
    version = '1.0.0'
  else:
    version = '0.1.0'
  return f'gs://gresearch/robotics/{dataset_name}/{version}'


def as_gif(images, path='temp.gif'):
  # Render the images as the gif:
  images[0].save(path, save_all=True, append_images=images[1:], duration=1000/15, loop=0)
  gif_bytes = open(path,'rb').read()
  return gif_bytes

if __name__ == "__main__" :
    dataset = "droid"
    builder = tfds.builder_from_directory(builder_dir=dataset2path(dataset))
    print(builder.info.features)
    display_key = 'exterior_image_1_left'
    if display_key not in builder.info.features['steps']['observation']:
        raise ValueError(
            f"The key {display_key} was not found in this dataset.\n"
            + "Please choose a different image key to display for this dataset.\n"
            + "Here is the observation spec:\n"
            + str(builder.info.features['steps']['observation']))
    ds = builder.as_dataset(split='train')
    episode = next(iter(ds))
    images = [step['observation'][display_key] for step in episode['steps']]
    images = [Image.fromarray(image.numpy()) for image in images]
    gif_bytes = as_gif(images)
    with open('temp_bridge.gif', 'wb') as f:
        f.write(gif_bytes)