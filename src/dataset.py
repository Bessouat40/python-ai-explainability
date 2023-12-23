import random
from numpy import concatenate, array

class Dataset:
    def __init__(self, train_images, test_images, val_images) :
        self.train_images, self.train_labels = train_images
        self.test_images, self.test_labels = test_images
        self.val_images, self.val_labels = val_images
        self.train_length = len(self.train_images)
        self.test_length = len(self.test_images)
        self.val_length = len(self.val_images)
        self.val_length = len(self.val_images)
        self.image_shape = self.train_images[0].shape

    def update_summary(self) :
        self.train_length = len(self.train_images)
        self.test_length = len(self.test_images)
        self.val_length = len(self.val_images)
        self.val_length = len(self.val_images)
        self.image_shape = self.train_images[0].shape

    def summary(self):
        print('-------------------- Dataset Summary --------------------\n')
        print('Number of train images : ', self.train_length)
        print('\n')
        print('Number of test images : ', self.test_length)
        print('\n')
        print('Number of validation images : ', self.val_length)
        print('\n')
        print('Shape of each images : ', self.image_shape)
        print('---------------------------------------------------------')

    @staticmethod
    def get_part_of_data(data, labels, quantity):
        idx = int(len(data) * quantity)
        new_data, new_labels = data[:idx], labels[:idx]
        return data[idx:], labels[idx:], new_data, new_labels

    def train_test_val_split(self, data, labels, training_quantity, test_quantity):
        _informations = list(zip(data, labels))
        random.shuffle(_informations)
        data, labels = zip(*_informations)
        _data, _labels, train_data, train_labels = self.get_part_of_data(list(data), list(labels), training_quantity)
        test_quantity = test_quantity / (1 - training_quantity)
        val_data, val_labels, test_data, test_labels = self.get_part_of_data(_data, _labels, test_quantity)
        return array(train_data), array(train_labels), array(test_data), array(test_labels), array(val_data), array(val_labels)


    @staticmethod
    def concat_data(*args) :
        _datas = concatenate([arg[0] for arg in args])
        _labels = concatenate([arg[1] for arg in args])
        return _datas, _labels

    def balance_data(self, training_quantity, test_quantity):
        _train = [self.train_images, self.train_labels]
        _test = [self.test_images, self.test_labels]
        _val = [self.val_images, self.val_labels]
        _data, _labels = self.concat_data(_train, _test, _val)
        self.train_images, self.train_labels, self.test_images, self.test_labels, self.val_images, self.val_labels = self.train_test_val_split(_data, _labels, training_quantity, test_quantity)
        self.update_summary()
