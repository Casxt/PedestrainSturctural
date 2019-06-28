import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
import sys
import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.models import load_model
from keras.preprocessing import image

from config import *

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6
session = tf.Session(config=config)
K.set_session(session)
model_path, img_path = sys.argv[1:]  # 'checkpoint/159.hdf5' 'test/090001.jpg'
model = load_model(model_path)

data = np.empty((1, *img_target_size))
img = image.load_img(img_path, target_size=img_target_size[0:2])
data[0] = image.img_to_array(img) / 255.0

predict_classes = model.predict(data, batch_size=1)
for pc in predict_classes:
    for k in classes.keys():
        if pc[classes[k]] > 0.5:
            print(k, round(pc[classes[k]], 4) * 100)
