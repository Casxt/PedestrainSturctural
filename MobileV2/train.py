import os
import sys
from pathlib import Path

sys.path.append(str(Path(os.path.abspath(os.path.dirname(__file__))).parent))
# 指定显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.applications.mobilenet_v2 import MobileNetV2

from tools import DataGenerator, loadMate, MultiLableAccuracy, MultiLableLoss, MultiLabelTensorBoard

img_target_size = (224, 224, 3)
batch_size = 32
classes, img_path, img_label = loadMate()

checkPointer = ModelCheckpoint(filepath="./MobileV2/checkpoint/{epoch:02d}.hdf5",
                               save_best_only=False, verbose=1, period=1)

tensorBoard = MultiLabelTensorBoard(log_dir='./MobileV2/logs',  # log 目录
                                    update_freq=1000,  # 按照何等频率（samples）来计算
                                    batch_size=batch_size,  # 用多大量的数据计算直方图
                                    write_graph=True,  # 是否存储网络结构图
                                    write_grads=True,  # 是否可视化梯度直方图
                                    write_images=True)  # 是否可视化参数

train = DataGenerator(img_path[0:80000], img_label[0:80000], classes, batch_size, img_target_size)
validation = DataGenerator(img_path[80000:90000], img_label[80000:90000], classes, batch_size, img_target_size)

base_model = MobileNetV2(weights=None, include_top=False, input_shape=img_target_size)  # 'imagenet'

x = base_model.output
print(x.shape)
x = GlobalAveragePooling2D()(x)
print(x.shape)
predictions = Dense(len(classes), activation='sigmoid')(x)
print(predictions.shape)
model = Model(inputs=base_model.input, outputs=predictions)

# 参见 https://stackoverflow.com/questions/42081257/keras-binary-crossentropy-vs-categorical-crossentropy-performance
# 和 https://stackoverflow.com/questions/45799474/keras-model-evaluate-vs-model-predict-accuracy-difference-in-multi-class-nlp-ta/45834857#45834857
model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              metrics=["categorical_accuracy", "mean_squared_error", "mean_absolute_error",
                       "mean_absolute_percentage_error", "categorical_crossentropy", MultiLableAccuracy,
                       MultiLableLoss])
model.fit_generator(generator=train, epochs=250, verbose=1, validation_data=validation,
                    class_weight=None, workers=4, initial_epoch=0, callbacks=[checkPointer, tensorBoard])
model.save("MobileV2.hdf5")
