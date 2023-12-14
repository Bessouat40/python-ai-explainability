import os
from PIL import Image 
from numpy import asarray, resize
from .dataset import Dataset

class Loader:
    def __init__(self, target_size) :
        self.path_to_data = os.getenv("PATH_TO_IMAGES") + '/'
        self.path_to_test = os.getenv("PATH_TO_TEST_IMAGES") + '/'
        self.target_size = target_size
        self.load_paths()
        self.load_data()
    
    def load_paths(self):
        _images_dir = os.listdir(self.path_to_data)
        _test_dir = os.listdir(self.path_to_test)
        self.images_dir = [self.path_to_data + path for path in _images_dir]
        self.test_dir = [self.path_to_test + path for path in _test_dir]

    def load_image(self, path) :
        try : return resize(asarray(Image.open(path)), self.target_size)/255.
        except : None
        
    def load_data(self) :
        _train_img = [self.load_image(image_dir) for image_dir in self.images_dir]
        _test_img = [self.load_image(image_dir) for image_dir in self.test_dir]

        self.train_img = [img for img in _train_img if img is not None]
        self.test_img = [img for img in _test_img if img is not None]

    def create_dataset(self) :
        return Dataset(self.train_img, self.test_img)
