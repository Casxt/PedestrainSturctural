import os
import sys
from keras.models import load_model
from tools import DataGenerator, loadMate

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
img_target_size = (224, 224, 3)
batch_size = 32
classes, img_path, img_label = loadMate()

validation = DataGenerator(img_path[90000:], img_label[90000:], classes, batch_size, img_target_size)

model_path = sys.argv[1]  # 'checkpoint/159.hdf5'
model = load_model(model_path)
model.summary()
score = model.evaluate_generator(generator=validation, verbose=1, steps=500, workers=5)
print("Loss: ", score[0], "Accuracy: ", score[1])