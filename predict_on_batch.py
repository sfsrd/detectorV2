import os
import numpy as np
import cv2
from keras.models import load_model
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices), ': ', physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)

X_data = []
files_dir = "/home/ariel/ADUK/cvml/ANYFOR/dataset_single_smoke/"
files = os.listdir(files_dir)

MODEL_FILE = '/home/ariel/ADUK/cvml/ANYFOR/model/classifier/TF_MODELS/whole image/model_ResNet152_25.model'
model = load_model(MODEL_FILE, compile = True)
model.summary()
image_size = 128

for my_file in files:
    img_path = str(files_dir) + str(my_file)
    print(img_path)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    image = np.array(image)
    X_data.append(image)

print('X_data shape:', np.array(X_data).shape)
X_data = np.array(X_data)

y_pred = model.predict(X_data, batch_size=6)
print('y_pred: ', y_pred)
print('y len: ', len(y_pred))