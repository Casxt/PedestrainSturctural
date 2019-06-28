import os
import sys
from pathlib import Path

import keras
from keras.engine.saving import load_model

sys.path.append(str(Path(os.path.abspath(os.path.dirname(__file__))).parent))
# 指定显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.applications.mobilenet_v2 import MobileNetV2
import keras.backend as K
from tools import DataGenerator, loadMate, MultiLableLoss, MultiLableAccuracy, MultiLabelTensorBoard, \
    MultiLableAccuracyLoss

img_target_size = (224, 224, 3)
batch_size = 32
classes, img_path, img_label = loadMate()

checkPointer = ModelCheckpoint(filepath="./MobileV2-MultiLabelLoss/checkpoint/{epoch:02d}.hdf5",
                               save_best_only=True, verbose=1, period=1)

tensorBoard = MultiLabelTensorBoard(log_dir='./MobileV2-MultiLabelLoss/logs',  # log 目录
                                    histogram_freq=0,
                                    update_freq=1000,  # 按照何等频率（samples）来计算
                                    batch_size=batch_size,  # 用多大量的数据计算直方图
                                    write_graph=True,  # 是否存储网络结构图
                                    write_grads=True,  # 是否可视化梯度直方图
                                    write_images=True)  # 是否可视化参数

train = DataGenerator(img_path[0:80000], img_label[0:80000], classes, batch_size, img_target_size)
validation = DataGenerator(img_path[80000:90000], img_label[80000:90000], classes, batch_size, img_target_size)

# base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=img_target_size)

# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# predictions = Dense(len(classes), activation='sigmoid')(x)
# model = Model(inputs=base_model.input, outputs=predictions)
model = load_model("/home/zhangkai/pedestrianstructuration/MobileV2-MultiLabelLoss/checkpoint/15.hdf5", compile=False)
# model = Model(inputs=base_model.input, outputs=base_model.output)
optimizers = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0001, amsgrad=False)
model.compile(optimizer=optimizers, loss=MultiLableAccuracyLoss,
              metrics=["categorical_accuracy", "mean_squared_error", "mean_absolute_error",
                       "mean_absolute_percentage_error", "categorical_crossentropy", MultiLableAccuracy, MultiLableLoss])
model.fit_generator(generator=train, epochs=250, verbose=1, validation_data=validation,
                    workers=4, initial_epoch=16, callbacks=[checkPointer, tensorBoard])
model.save("MobileV2-MultiLabelLoss.hdf5")
