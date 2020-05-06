from tensorflow.keras.layers import UpSampling2D, Conv2D, BatchNormalization, Dense, \
    concatenate, Reshape, Input, Flatten, ZeroPadding2D, Activation, Lambda, add, LeakyReLU, ReLU
from tensorflow.keras.backend import exp, random_normal, tile, expand_dims, constant
from tensorflow.keras import Model

def upSampleBlock(X, filters):
    X = UpSampling2D((2,2))(X)
    X = Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False)(X)
    X = BatchNormalization()(X)
    X = ReLU()(X)
    return X

def residualBlock(inp, size):
    X = Conv2D(size, kernel_size=3, padding='same', strides=1)(inp)
    X = BatchNormalization()(X)
    X = ReLU()(X)
    X = Conv2D(size, kernel_size=3, padding='same', strides=1)(X)
    X = BatchNormalization()(X)
    X = add([X, inp])
    return ReLU()(X)

def conv3(X, size, padding=True):
    if padding:
        return Conv2D(size, kernel_size=3, strides=1, padding='same', use_bias=False)(X)
    return Conv2D(size, kernel_size=3, strides=1, use_bias=False)(X)

def combine(inputs):
    c = inputs[0]
    x = inputs[1]

    c = expand_dims(c, axis=1)
    c = expand_dims(c, axis=1)
    c = tile(c, [1, 16, 16, 1])
    return concatenate([c, x], axis=3)

def samplec(x):
    mean = x[:, :128]
    log_sigma = x[:, 128:]

    stddev = exp(log_sigma)
    epsilon = random_normal(shape=constant((mean.shape[1],), dtype='int32'))
    c = stddev * epsilon + mean
    return c

def generator():

    inp_text = Input(shape=(1024,))
    inp_image = Input(shape=(64, 64, 3))

    text = Dense(256)(inp_text)
    text = LeakyReLU(.2)(text)
    X1 = Lambda(samplec)(text)

    X2 = ZeroPadding2D(padding=(1, 1))(inp_image)
    X2 = conv3(X2, 128, False)
    X2 = ReLU()(X2)
    X2 = ZeroPadding2D(padding=(1, 1))(X2)
    X2 = Conv2D(256, kernel_size=4, strides=2, use_bias=False)(X2)
    X2 = BatchNormalization()(X2)
    X2 = ReLU()(X2)
    X2 = ZeroPadding2D(padding=(1, 1))(X2)
    X2 = Conv2D(512, kernel_size=4, strides=2, use_bias=False)(X2)
    X2 = BatchNormalization()(X2)
    X2 = ReLU()(X2)
    X = Lambda(combine)([X1, X2])

    X = ZeroPadding2D(padding=(1, 1))(X)
    X = Conv2D(512, kernel_size=3, strides=1, use_bias=False)(X)
    X = BatchNormalization()(X)
    X = ReLU()(X)

    X = residualBlock(X, 512)

    X = upSampleBlock(X, 512)
    X = upSampleBlock(X, 256)
    X = upSampleBlock(X, 128)
    X = upSampleBlock(X, 64)

    X = Conv2D(3, kernel_size=3, strides=1, padding='same', use_bias=False)(X)
    image = Activation('tanh')(X)

    model = Model(inputs=[inp_text, inp_image], outputs=[image, text])
    return model


def discrim_conv(X, size):
    return Conv2D(size, kernel_size=4, strides=2, padding='same', use_bias=False)(X)


def discrim_conv_block(X, size, alpha=.2):
    X = discrim_conv(X, size)
    X = BatchNormalization()(X)
    X = LeakyReLU(alpha)(X)
    return X


def conv1(X, size):
    return Conv2D(size, kernel_size=1, strides=1, padding='same', use_bias=False)(X)


def discriminator():
    inp = Input(shape=[256, 256, 3])
    inp2 = Input(shape=(4, 4, 128))

    X = discrim_conv(inp, 64)
    X = LeakyReLU(.2)(X)
    X = discrim_conv_block(X, 128, .2)
    X = discrim_conv_block(X, 256, .2)
    X = discrim_conv_block(X, 512, .2)
    X = discrim_conv_block(X, 1024, .2)
    X = discrim_conv_block(X, 2048, .2)
    X = conv1(X, 1024)
    X = BatchNormalization()(X)
    X = LeakyReLU(.2)(X)
    X = conv1(X, 512)
    X = BatchNormalization()(X)

    X2 = conv1(X, 128)
    X2 = BatchNormalization()(X2)
    X2 = LeakyReLU(.2)(X2)
    X2 = conv3(X2, 128)
    X2 = BatchNormalization()(X2)
    X2 = LeakyReLU(.2)(X2)
    X2 = conv3(X2, 512)
    X2 = BatchNormalization()(X2)
    X_comb = add([X, X2])
    X_comb = LeakyReLU(.2)(X_comb)

    X_c = concatenate([X_comb, inp2])

    X_out  = conv1(X_c, 512)
    X_out = BatchNormalization()(X_out)
    X_out = LeakyReLU(.2)(X_out)
    X_out = Flatten()(X_out)
    X_out = Dense(1)(X_out)
    X_out = Activation('sigmoid')(X_out)

    model = Model(inputs=[inp, inp2], outputs=[X_out])
    return model

if __name__ == "__main__":
    print("setup")
