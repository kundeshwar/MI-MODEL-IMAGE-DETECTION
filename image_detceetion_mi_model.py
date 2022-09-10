from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
import cv2


from keras.applications import inception_v3
model = inception_v3.InceptionV3(weights="imagenet")
model.summary()

from PIL import Image
image = Image.open('mobile.jpg')
plt.imshow(image)

from keras import preprocessing
img = preprocessing.image.load_img('mobile.jpg', target_size = (299, 299))
x = preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = inception_v3.preprocess_input(x)
prediction = model.predict(x)
labels = inception_v3.decode_predictions(prediction, top=3)[0]
labels
