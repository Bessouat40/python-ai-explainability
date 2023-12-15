from tensorflow.keras import optimizers

class Trainer:
    def __init__(self, model, dataset, epochs = 25, batch_size = 128, lr = 0.0001, momentum = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = model
        self.dataset = dataset

    def compile(self):
        self.model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=self.lr, momentum=self.momentum), metrics=["accuracy"])

    def train(self):
        self.model.fit(self.dataset.train_images, self.dataset.train_labels,
          epochs=self.epochs,
          batch_size=self.batch_size,
          shuffle=True,
          validation_data=(self.dataset.val_images, self.dataset.val_labels))