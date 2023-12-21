import os
from PIL import Image 
from numpy import asarray, resize, concatenate, stack
from dataset import Dataset

class Loader:
    def __init__(self, target_size) :
        self.path_train_data = os.getenv("PATH_TO_IMAGES") + '/'
        self.path_train_data2 = os.getenv("PATH_TO_IMAGES2") + '/'
        self.path_test_data = os.getenv("PATH_TO_TEST_IMAGES") + '/'
        self.path_test_data2 = os.getenv("PATH_TO_TEST_IMAGES2") + '/'
        self.path_val_data = os.getenv("PATH_TO_VALIDATION_IMAGES") + '/'
        self.path_val_data2 = os.getenv("PATH_TO_VALIDATION_IMAGES2") + '/'
        self.target_size = target_size
        self.load_images()

    def process_data(self, data_path, label) :
        _dirs = os.listdir(data_path)
        _list_dirs = [data_path + path for path in _dirs]
        _imgs, _labels = self.load_images_labels(_list_dirs, label)
        _imgs = [img for img in _imgs if img is not None]
        return _imgs, _labels
    
    def load_data(self, path1, path2):
        _data, _labels = self.process_data(path1, 0)
        _data2, _labels2 = self.process_data(path2, 1)
        return [concatenate((_data, _data2), axis=0), concatenate((_labels, _labels2), axis=0)]

    def load_images(self):
        self.train_data = self.load_data(self.path_train_data, self.path_train_data2)
        self.test_data = self.load_data(self.path_test_data, self.path_test_data2)
        self.val_data = self.load_data(self.path_val_data, self.path_val_data2)


    def load_image(self, path) :
        try:
            img = Image.open(path)
            img_resized = resize(asarray(img), self.target_size)

            # Convert to three channels by stacking the grayscale image
            img_stack = stack((img_resized,)*3, axis=-1)
            return img_stack / 255.0
        except : None
        
    def load_images_labels(self, data_dir, label) :
        _images = [self.load_image(image_dir) for image_dir in data_dir]
        _labels = [label for i in range(len(_images))]
        return [_images, _labels]

    def create_dataset(self) :
        return Dataset(self.train_data, self.test_data, self.val_data)
