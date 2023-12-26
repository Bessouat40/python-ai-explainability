import scipy.ndimage as sp
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Model

def plot_activation(model, img):
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