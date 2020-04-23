import tensorflow as tf
from keras.models import Sequential
from keras.layers import UpSampling2D, Conv2D, BatchNormalization, Dense, Concatenate, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.backend import exp, random_normal, tile
from keras.activations import relu, tanh

def upblock(X, filters):
    X = UpSampling2D((2,2), interpolation='nearest')(X)
    X = Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False)(X)
    X = BatchNormalization()(X)
    X = relu(X)
    return X

def residual(inp, size):
    X = Conv2D(size, kernel_size=3, padding='same', strides=1)(inp)
    X = BatchNormalization()(X)
    X = relu(X)
    X = Conv2D(size, kernel_size=3, padding='same', strides=1)(X)
    X = BatchNormalization()(X)
    return relu(X + inp)

def conv3(X, size):
    return Conv2D(size, kernel_size=3, strides=1, padding='same', use_bias=False)(X)

def generator(text, images):

    #Text Embedding
    text = Dense(256)(text)
    text = relu(text)
    mu = text[:, :128]
    logvar = text[:, 128:]
    temp = exp(logvar)
    eps = random_normal(shape=mu.shape[1], dtype='int32')
    X2 = eps * temp + mu

    # Image Encoder
    X = conv3(images, 128)
    X = relu(X)
    X = Conv2D(256, kernel_size=4, strides=2, padding=1, use_bias=False)(X)
    X = BatchNormalization()(X)
    X = relu(X)
    X = Conv2D(512, kernel_size=4, strides=2, padding=1, use_bias=False)(X)
    X = BatchNormalization()(X)
    X = relu(X)

    # Combine Image and Embedding
    X2 = Reshape([-1, 128, 1, 1])(X2)
    X2 = tile(X2, [1,1,16,16])
    X = Concatenate([X, X2], axis=1)
    X = Conv2D(512, kernel_size=3, strides=1, padding=1, use_bias=False)(X)
    X = BatchNormalization()(X)
    X = relu(X)

    # Residual Layers
    for i in range(4):
        X = residual(X, 512)

    # Upsampling Layers
    X = upblock(X, 512)
    X = upblock(X, 256)
    X = upblock(X, 128)
    X = upblock(X, 64)

    X = Conv2D(3, kernel_size=3, strides=1, padding='same', use_bias=False)(X)
    image = tanh(X)
    return image

def discriminator(X):
    print("discriminator")

if __name__ == "__main__":
    print("setup")
