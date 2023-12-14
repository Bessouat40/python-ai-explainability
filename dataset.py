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