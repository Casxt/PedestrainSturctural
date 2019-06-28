import keras.backend as K
from keras.losses import categorical_crossentropy, binary_crossentropy
import tensorflow as tf


def MultiLableAccuracy(y_true, y_pred):
    # 使用0.5 作为阈值判别预测值至0-1
    y_pred = tf.divide(tf.add(tf.sign(tf.add(y_pred, -0.5)), 1), 2)
    return K.sum(tf.cast(K.equal(y_pred, y_true), dtype=tf.float32), axis=-1)


def MultiLableAccuracyLoss(y_true, y_pred):
    # 使用0.5 作为阈值判别预测值至0-1
    y_pred = tf.tanh(tf.add(y_pred, -0.5))
    return K.sum(tf.abs(tf.subtract(y_pred, y_true)))


def MultiLableLoss(y_true, y_pred):
    def repack(t):
        gender_male, age, direction, others, gender_female = tf.split(t, [1, 3, 3, 19, 1], -1)
        gender = tf.concat([gender_male, gender_female], -1)
        return gender, age, direction, others

    print(y_pred.shape)
    gender_pred, age_pred, direction_pred, others_pred = repack(y_pred)
    gender_true, age_true, direction_true, others_true = repack(y_true)
    print(gender_pred.shape, age_pred.shape, direction_pred.shape, others_pred.shape)
    print(gender_true.shape, age_true.shape, direction_true.shape, others_true.shape)
    return tf.add_n([categorical_crossentropy(gender_true, gender_pred),
                     categorical_crossentropy(age_true, age_pred),
                     categorical_crossentropy(direction_true, direction_pred),
                     binary_crossentropy(others_true, others_pred)])
