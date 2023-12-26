from src.loader import Loader
from matplotlib import pyplot
import scipy.ndimage as sp
import matplotlib.pyplot as plt
import numpy as np
import dotenv

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Conv2D, Lambda, GlobalAveragePooling2D, Dense
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

def plot_activation(img):
    pred = model.predict(img[np.newaxis,:,:,:])
    pred_class = int(pred > 0.5)
    print('pred : ', pred_class)
    weights = model.layers[-1].get_weights()[0]
    class_weights = weights[:, 0]
    intermediate = Model(model.input,
                         model.get_layer("block5_conv3").output)
    conv_output = intermediate.predict(img[np.newaxis,:,:,:])
    conv_output = np.squeeze(conv_output)
    h = int(img.shape[0]/conv_output.shape[0])
    w = int(img.shape[1]/conv_output.shape[1])
    act_maps = sp.zoom(conv_output, (h, w, 1), order=1)
    out = np.dot(act_maps.reshape((img.shape[0]*img.shape[1],512)), 
                 class_weights).reshape(img.shape[0],img.shape[1])
    plt.imshow(img.astype('float32').reshape(img.shape[0],
               img.shape[1],3))
    plt.imshow(out, cmap='jet', alpha=0.35)
    plt.title('Pneumonia' if pred_class == 1 else 'No Pneumonia')

plot_activation(dataset.test_images[6])
print(dataset.test_labels[6])