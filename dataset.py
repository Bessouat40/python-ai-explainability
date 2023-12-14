class Dataset:
    def __init__(self, train_images, test_images) :
        self.train_images = train_images
        self.test_images = test_images
        self.length = len(train_images)
        self.image_shape = train_images[0].shape