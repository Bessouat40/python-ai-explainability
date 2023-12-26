from src.loader import Loader
from utils.generics_functions import plot_activation
import dotenv

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import optimizers
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

dotenv.load_dotenv()

loader = Loader((224, 224))

dataset = loader.create_dataset()
dataset.summary()

#Balance dataset for better training
dataset.balance_data(0.8, 0.10)
dataset.summary()

vgg_conv = VGG16(weights='./vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=(224, 224, 3))

for layer in vgg_conv.layers[:-8]:
    layer.trainable = False

x = vgg_conv.output
x = GlobalAveragePooling2D()(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(vgg_conv.input, x)
model.compile(loss = "binary_crossentropy", optimizer = optimizers.legacy.SGD(learning_rate=0.005, momentum=0.9), metrics=["accuracy"])

checkpoint_path = "model_checkpoints/checkpoint-{epoch:02d}-{val_accuracy:.2f}.h5"
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False,  
    monitor='val_accuracy',   
    mode='max',               
    save_best_only=True,      
    verbose=1
)

model.fit(dataset.train_images, dataset.train_labels,
          epochs=25,
          batch_size=128,
          shuffle=True,
          validation_data=(dataset.val_images, dataset.val_labels),
          callbacks=[checkpoint_callback])

model.evaluate(dataset.test_images, dataset.test_labels, verbose=2)

model.save('model_miniforge')

plot_activation(model, dataset.test_images[6])
print(dataset.test_labels[6])