import os
from PIL import Image 
from numpy import asarray, resize
from dataset import Dataset
from numpy import concatenate, stack

class Loader:
    def __init__(self, target_size) :
        self.path_train_data = os.getenv("PATH_TO_IMAGES") + '/'
        self.path_train_data2 = os.getenv("PATH_TO_IMAGES2") + '/'
        self.path_test_data = os.getenv("PATH_TO_TEST_IMAGES") + '/'
        self.path_test_data2 = os.getenv("PATH_TO_TEST_IMAGES2") + '/'
        self.path_val_data = os.getenv("PATH_TO_VALIDATION_IMAGES") + '/'
        self.path_val_data2 = os.getenv("PATH_TO_VALIDATION_IMAGES2") + '/'
        self.target_size = target_size
        self.load_paths()
        self.load_images()
    
    def load_paths(self):
        _train_dir = os.listdir(self.path_train_data)
        _train_dir2 = os.listdir(self.path_train_data2)
        _test_dir = os.listdir(self.path_test_data)
        _test_dir2 = os.listdir(self.path_test_data2)
        _val_dir = os.listdir(self.path_val_data)
        _val_dir2 = os.listdir(self.path_val_data2)

        self.train_dir = [self.path_train_data + path for path in _train_dir]
        self.train_dir2 = [self.path_train_data2 + path for path in _train_dir2]
        self.test_dir = [self.path_test_data + path for path in _test_dir]
        self.test_dir2 = [self.path_test_data2 + path for path in _test_dir2]
        self.val_dir = [self.path_val_data + path for path in _val_dir]
        self.val_dir2 = [self.path_val_data2 + path for path in _val_dir2]

    def load_images(self):
        _train_img, _train_labels = self.load_data(self.train_dir, 0)
        _train_img2, _train_labels2 = self.load_data(self.train_dir2, 1)
        _test_img, _test_labels = self.load_data(self.test_dir, 0)
        _test_img2, _test_labels2 = self.load_data(self.test_dir2, 1)
        _val_img, _val_labels = self.load_data(self.val_dir, 0)
        _val_img2, _val_labels2 = self.load_data(self.val_dir2, 1)
        
        _train_img = [img for img in _train_img if img is not None]
        _train_img2 = [img for img in _train_img2 if img is not None]
        _test_img = [img for img in _test_img if img is not None]
        _test_img2 = [img for img in _test_img2 if img is not None]
        _val_img = [img for img in _val_img if img is not None]
        _val_img2 = [img for img in _val_img2 if img is not None]


        self.train_data = [concatenate((_train_img, _train_img2), axis=0), concatenate((_train_labels, _train_labels2), axis=0)]
        self.test_data = [concatenate((_test_img, _test_img2), axis=0), concatenate((_test_labels, _test_labels2), axis=0)]
        self.val_data = [concatenate((_val_img, _val_img2), axis=0), concatenate((_val_labels, _val_labels2), axis=0)]


    def load_image(self, path) :
        try:
            img = Image.open(path)
            img_resized = resize(asarray(img), self.target_size)

            # Convert to three channels by stacking the grayscale image
            img_stack = stack((img_resized,)*3, axis=-1)
            return img_stack / 255.0
        except : None
        
    def load_data(self, data_dir, label) :
        _images = [self.load_image(image_dir) for image_dir in data_dir]
        _labels = [label for i in range(len(_images))]
        return [_images, _labels]

    def create_dataset(self) :
        return Dataset(self.train_data, self.test_data, self.val_data)
