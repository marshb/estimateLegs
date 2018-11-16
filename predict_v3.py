from keras.models import Model, load_model
from scipy import misc, ndimage
from keras import backend as K
from keras.layers import *
from keras.optimizers import Adam
import os
from keras.losses import binary_crossentropy
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
import cv2
from tqdm import tqdm

# Constants
HEIGHT = 1024
WIDTH = 512
CHANNELS = 1

def down(filters, input_):
    down_ = Conv2D(filters, (3, 3), padding='same')(input_)
    down_ = BatchNormalization(epsilon=1e-4)(down_)
    down_ = Activation('relu')(down_)
    down_ = Conv2D(filters, (3, 3), padding='same')(down_)
    down_ = BatchNormalization(epsilon=1e-4)(down_)
    down_res = Activation('relu')(down_)
    down_pool = MaxPooling2D((2, 2), strides=(2, 2))(down_)
    return down_pool, down_res


def up(filters, input_, down_):
    up_ = UpSampling2D((2, 2))(input_)
    up_ = concatenate([down_, up_], axis=3)
    up_ = Conv2D(filters, (3, 3), padding='same')(up_)
    up_ = BatchNormalization(epsilon=1e-4)(up_)
    up_ = Activation('relu')(up_)
    up_ = Conv2D(filters, (3, 3), padding='same')(up_)
    up_ = BatchNormalization(epsilon=1e-4)(up_)
    # up_ = Activation('relu')(up_)
    # up_ = Conv2D(filters, (3, 3), padding='same')(up_)
    # up_ = BatchNormalization(epsilon=1e-4)(up_)
    up_ = Activation('relu')(up_)
    return up_


def get_unet_1024(input_shape=(WIDTH, HEIGHT, CHANNELS), num_classes=1):
# def get_unet_1024(input_shape=(HEIGHT, WIDTH, CHANNELS), num_classes=1):
    inputs = Input(shape=input_shape)

    # down0b, down0b_res = down(8, inputs)
    down0a, down0a_res = down(16, inputs)
    down0, down0_res = down(32, down0a)
    down1, down1_res = down(64, down0)
    down2, down2_res = down(128, down1)
    down3, down3_res = down(256, down2)
    down4, down4_res = down(512, down3)

    center = Conv2D(512, (3, 3), padding='same')(down4)
    center = BatchNormalization(epsilon=1e-4)(center)
    center = Activation('relu')(center)
    center = Conv2D(512, (3, 3), padding='same')(center)
    center = BatchNormalization(epsilon=1e-4)(center)
    center = Activation('relu')(center)

    up4 = up(512, center, down4_res)
    up3 = up(256, up4, down3_res)
    up2 = up(128, up3, down2_res)
    up1 = up(64, up2, down1_res)
    up0 = up(32, up1, down0_res)
    up0a = up(16, up0, down0a_res)
    # up0b = up(8, up0a, down0b_res)

    #final_conv1 = Conv2D(16, (3, 3), padding='same', activation='relu', name='final_conv1')(up0a)
    classify = Conv2D(num_classes, (1, 1), activation='sigmoid', name='final_layer')(up0a)

    model = Model(inputs=inputs, outputs=classify)

    return model


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)

model_path = "./models3"
model = get_unet_1024()
model.load_weights(os.path.join(model_path, "model-1541128407-weights.h5"))
model.compile(loss=bce_dice_loss, optimizer=Adam(1e-5), metrics=[dice_coef])

model = load_model(os.path.join(model_path, 'model-1541128407.h5'),
                   custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_coef': dice_coef})


test_data_dir = './data/test'
results_dir = 'test_results'
fileslist = os.listdir(test_data_dir)

for fn in tqdm(fileslist):
    img_path = os.path.join(test_data_dir, fn)
    img = misc.imread(img_path)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
    img = np.expand_dims(img, axis=0)
    out_predict = model.predict(img)
    out_predict = np.squeeze(out_predict, axis=0)
    out_predict[out_predict > 0.5] = 255
    path_saved = os.path.join(results_dir, fn)
    cv2.imwrite(path_saved, out_predict)


"""
img_path = "./data/test/000004.bmp"
img_ = misc.imread(img_path)
# img_ = np.transpose(img_, (1, 0))
print("img_ shape:", img_.shape)

# size = (512, 512)
# img_ = cv2.resize(img, size)
plt.imshow(img_)
plt.show()

img_ = np.expand_dims(img_, axis=2)
img_ = np.expand_dims(img_, axis=0)

# print("input shape:", img_.shape)
img_out = model.predict(img_)
# print("output shape", img_out.shape)
img_out = np.squeeze(img_out, axis=0)
# print("output shape", img_out.shape)
img_out[img_out > 0.5] = 255

# save results
results_dir = './test_results'
dir, test_ID = os.path.split(img_path)
path_saved = os.path.join(results_dir, test_ID)
cv2.imwrite(path_saved, img_out)
"""