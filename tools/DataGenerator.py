import numpy as np
from keras.preprocessing import image
from keras.utils import Sequence

class DataGenerator(Sequence):
    """Generates data for Keras"""

    def __init__(self, images, labels, classes, batch_size, img_size):
        """
        images should be path of img
        labels should be set of attribute in string format
        classes should be in dict[classNum]:orderNum format
        batch_size is batch_size like 32
        img_size is ths size of img like (32, 32, 3)
        """
        self.img_size = img_size
        self.batch_size = batch_size
        self.labels = labels
        # zip paths, img_labels to images
        self.images = list(zip(images, labels))
        self.classes = classes

        np.random.shuffle(self.images)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        img_batch = self.images[index *
                                self.batch_size:(index + 1) * self.batch_size]

        # unzip images to paths, img_labels
        paths, img_labels = zip(*img_batch)

        # Generate data
        data = np.empty((self.batch_size, *self.img_size))
        label = np.zeros((self.batch_size, len(self.classes)), dtype="float32")

        for index, path in enumerate(paths):
            # 设置图像
            img = image.load_img(path, target_size=self.img_size[0:2])
            data[index] = image.img_to_array(img) / 255.0
            # 设置label
            for label_name in img_labels[index]:
                label[index][self.classes[label_name]] = 1.0

        return data, label

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        np.random.shuffle(self.images)

    def __iter__(self):
        """Creates an infinite generator that iterate over the Sequence.

        Yields:
          Sequence items.
        """
        while True:
            for item in (self[i] for i in range(len(self))):
                yield item
