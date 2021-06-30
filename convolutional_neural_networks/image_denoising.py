# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <markdowncell>

# # Image Denoising Challenge
# 
# The goal for this challenge is to leverage your knowledge of Deep Learning to design and train a denoising model. For a given noisy image $X$, our model should learn to predict the denoised image $y$.
# 
# 
# **Objectives**
# - Visualize images
# - Preprocess images for the neural network
# - Fit a custom CNN for the task

# <markdowncell>

# ## 1. Load Data
# 
# 👉 Let's download the dataset archive.
# It contains RGB and Black & White images we will be using for the rest of this challenge.

# <codecell>

#! curl https://wagon-public-datasets.s3.amazonaws.com/certification_france_2021_q2/paintings.zip > paintings.zip
#! unzip -nq "paintings.zip" 
#! rm "paintings.zip"
#! ls -l

# <codecell>

import matplotlib.pyplot as plt
import numpy as np
import glob
import tensorflow as tf

from PIL import Image
from collections import Counter
from tqdm import tqdm
from tensorflow.image import resize


# <codecell>

dataset_paths = glob.glob("./paintings/*.jpg")
dataset_paths

# <markdowncell>

# ❓ **Display the image at index `53` of this dataset_paths (i.e the 54-th image)**
# 
# <details>
#     <summary>Hint</summary>
#     Use the <code>PIL.Image.open</code> and <code>matplotlib.pyplot.imshow</code> functions.
# </details>

# <codecell>

im = Image.open(dataset_paths[53])
plt.imshow(im);

# <markdowncell>

# ❓ **What is the shape of the image you displayed above `img_shape`?  How many dimensions `img_dim` does it have ?**

# <codecell>

print(f'type : {type(im)}, im_size : {im._size} , numpy shape : {np.array(im).shape}')
# numpy takes the height first, then the width

# <codecell>

im_np = np.array(im)
img_shape = im_np.shape
img_shape

# <codecell>

img_dim = im_np.ndim
img_dim

# <markdowncell>

# ❓ **What was in the image above?**

# <codecell>

img_shape = img_shape
img_dim = img_dim

# Uncomment the correct answer

is_portrait = True
#is_portrait = False

is_colored_image = True
#is_colored_image = False

# <codecell>

from nbresult import ChallengeResult
result = ChallengeResult(
    'data_loading',
    img_shape=img_shape,
    img_dim=img_dim,
    is_portrait=is_portrait,
    is_colored_image=is_colored_image
)

result.write()

# <markdowncell>

# ## 2. Processing

# <markdowncell>

# ❓ **Store all images from the dataset folder in a list of numpy arrays called `dataset_images`**
# 
# - It can take a while
# - If the dataset is too big to fit in memory, just take the first half (or quarter) of all pictures

# <codecell>

len(dataset_paths)

# <codecell>

dataset_images = [np.array(Image.open(path)) for path in dataset_paths]

# <codecell>

len(dataset_images), dataset_images[53].shape

# <markdowncell>

# ### 2.1 Reshape, Resize, Rescale
# 
# Let's simplify our dataset and convert it to a single numpy array

# <markdowncell>

# ❓ **First, check if that all the images in the dataset have the same number of dimensions**.
# - What do you notice?
# - How do you explain it? 

# <codecell>

image_dimensions = [img.ndim for img in dataset_images]
image_dimensions[:10], image_dimensions[:10]

# <codecell>

Counter([img.ndim for img in dataset_images])

# <codecell>

"We might have 72 black and white pictures, with only 1 channel in the last dim"

# <markdowncell>

# 👉 We convert for you all black & white images into 3-colored ones by duplicating the image on three channels, so as to have only 3D arrays

# <codecell>

dataset_images = [x if x.ndim==3 else np.repeat(x[:,:,None], 3, axis=2) for x in tqdm(dataset_images)]
set([x.ndim for x in dataset_images])

# <markdowncell>

# ❓ **What about their shape now ?**
# - Do they all have the same width/heights ? If not:
# - Resize the images (120 pixels height and 100 pixels width) in the dataset, using `tensorflow.image.resize` function.
# - Now that they all have the same shape, store them as a numpy array `dataset_resized`.
# - This array should thus be of size $(n_{images}, 120, 100, 3)$

# <codecell>

Counter([img.shape for img in dataset_images])

# <codecell>

dataset_resized = [resize(image, [120,100]) for image in dataset_images]

# <codecell>

len(dataset_resized), dataset_resized[0]

# <markdowncell>

# ❓ **Rescale the data of each image between $0$ and $1$**
# - Save your resulting list as `dataset_scaled`

# <codecell>

dataset_scaled =  [image/255. for image in dataset_resized]
dataset_scaled[0]

# <codecell>

np.max([tf.math.reduce_max(img).numpy() for img in dataset_scaled])

# <codecell>

np.min([tf.math.reduce_min(img).numpy() for img in dataset_scaled])

# <codecell>

print(type(dataset_scaled))
dataset_scaled = np.array(dataset_scaled)
print(type(dataset_scaled))

# <markdowncell>

# 👉 Now, we'll add for you some **random noise** to our images to simulate noise (that our model will try to remove later)

# <codecell>

NOISE_LEVEL = 0.2

dataset_noisy = np.clip(
    dataset_scaled + np.random.normal(
        loc=0,
        scale=NOISE_LEVEL,
        size=dataset_scaled.shape
    ).astype(np.float32),
    0,
    1
)
dataset_noisy.shape

# <markdowncell>

# ❓ **Plot a noisy image below to visualize the noise and compare it with the normal one**

# <codecell>

plt.imshow(dataset_noisy[53])

# <markdowncell>

# ❓ **Create your `(X_train, Y_train)`, `(X_test, Y_test)` training set for your problem**
# 
# - Remember you are trying to use "noisy" pictures in order to predict the "normal" ones.
# - Keeping about `20%` of randomly sampled data as test set

# <codecell>

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(dataset_noisy, dataset_scaled, test_size=0.2)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

# <codecell>

plt.imshow(dataset_noisy[530])

# <codecell>

plt.imshow(dataset_scaled[530])

# <codecell>

from nbresult import ChallengeResult
result = ChallengeResult(
    "preprocessing",
    X_train_shape = X_train.shape,
    Y_train_shape = Y_train.shape,
    X_std = X_train[:,:,:,0].std(),
    Y_std = Y_train[:,:,:,0].std(),
    first_image = Y_train[0]
)
result.write()

# <markdowncell>

# ## 3. Convolutional Neural Network
# 
# A commonly used neural network architecture for image denoising is the __AutoEncoder__.
# 
# <img src='https://github.com/lewagon/data-images/blob/master/DL/autoencoder.png?raw=true'>
# 
# Its goal is to learn a compact representation of your data to reconstruct them as precisely as possible.  
# The loss for such model must incentivize it to have __an output as close to the input as possible__.
# 
# For this challenge, __you will only be asked to code the Encoder part of the network__, since building a Decoder leverages layers architectures you are not familiar with (yet).

# <markdowncell>

# 👉 Run this code below if you haven't managed to build your own (X,Y) training sets. This will load them as solution
# 
# ```python
# ! curl https://wagon-public-datasets.s3.amazonaws.com/certification_france_2021_q2/data_painting_solution.pickle > data_painting_solution.pickle
# 
# import pickle
# with open("data_painting_solution.pickle", "rb") as file:
#     (X_train, Y_train, X_test, Y_test) = pickle.load(file)
#     
# ! rm data_painting_solution.pickle
# ```

# <markdowncell>

# ### 3.1 Architecture

# <markdowncell>

# 👉 Run the cell below that defines the decoder

# <codecell>

import tensorflow as tf
from tensorflow.keras import layers, losses, Sequential

# <codecell>

# We choose to compress images into a latent_dimension of size 6000
latent_dimensions = 6000

# We build a decoder that takes 1D-vectors of size 6000 to reconstruct images of shape (120,100,3)
decoder = Sequential(name='decoder')
decoder.add(layers.Reshape((30, 25, 8), input_dim=latent_dimensions))
decoder.add(layers.Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding="same", activation="relu"))
decoder.add(layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding="same", activation="relu"))
decoder.add(layers.Conv2D(filters=3, kernel_size=3, padding="same", activation="sigmoid"))
decoder.summary()

# <markdowncell>

# ❓ **Now, build the `encoder` that plugs correctly with the decoder defined above**. Make sure that:
# - The output of your `encoder` is the same shape as the input of the `decoder`
# - Use a convolutional neural network architecture without transfer learning
# - Keep it simple
# - Print model summary

# <codecell>

# CODE HERE YOUR ENCODER ARCHITECTURE AND PRINT IT'S MODEL SUMMARY

encoder = None

# <markdowncell>

# 👉 **Test your encoder below**

# <codecell>

# HERE WE BUILD THE AUTO-ENCODER (ENCODER + DECODER) FOR YOU. IT SHOULD PRINT A NICE SUMMARY
from tensorflow.keras.models import Model

x = layers.Input(shape=(120, 100, 3))
autoencoder = Model(x, decoder(encoder(x)), name="autoencoder")
autoencoder.summary()

# <markdowncell>

# ### 3.2 Training

# <markdowncell>

# ❓ **Before training the autoencoder, evaluate your baseline score**
# - We will use the mean absolute error in this challenge
# - Compute the baseline score on your test set in the "stupid" case where you don't manage to de-noise anything at all.
# - Store the result under `score_baseline`

# <codecell>

# YOUR CODE HERE

# <markdowncell>

# ❓ Now, **train your autoencoder**
# 
# - Use an appropriate loss
# - Adapt the learning rate of your optimizer if convergence is too slow/fast
# - Make sure your model does not overfit with appropriate control techniques
# 
# 💡 You will not be judged by the computing power of your computer, you can reach decent performance in less than 5 minutes of training without GPUs.

# <codecell>

# YOUR CODE HERE

# <markdowncell>

# ❓ **Plot your training and validation loss at each epoch using the cell below**

# <codecell>

# Plot below your train/val loss history
# YOUR CODE HERE
# YOUR CODE HERE
# YOUR CODE HERE


# Run also this code to save figure as jpg in path below (it's your job to ensure it works)
fig = plt.gcf()
plt.savefig("tests/history.png")

# <markdowncell>

# ❓ **Evaluate your performances on test set**
# - Compute your de-noised test set `Y_pred` 
# - Store your test score as `score_test`
# - Plot a de-noised image from your test set and compare it with the original and noisy one using the cell below

# <codecell>

# YOUR CODE HERE

# <codecell>

# RUN THIS CELL TO CHECK YOUR RESULTS
idx = 0

fig, axs = plt.subplots(1,3, figsize=(10,5))
axs[0].imshow(Y_test[idx])
axs[0].set_title("Clean image.")

axs[1].imshow(X_test[idx])
axs[1].set_title("Noisy image.")

axs[2].imshow(Y_pred[idx])
axs[2].set_title("Prediction.")

# Run this to save your results for correction
plt.savefig('tests/image_denoised.png')

# <markdowncell>

# 🧪 **Send your results below**

# <codecell>

from nbresult import ChallengeResult

result = ChallengeResult(
    "network",
    input_shape = list(encoder.input.shape),
    output_shape = list(encoder.output.shape),
    layer_names = [layer.name for layer in encoder.layers],
    trainable_params = sum([tf.size(w_matrix).numpy() for w_matrix in encoder.trainable_variables]),
    score_baseline = score_baseline,
    score_test = score_test,
)
result.write()
