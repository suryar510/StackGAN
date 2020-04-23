def downblock(x, out):
    x = Conv2D(out, (4, 4), padding='same', strides=2, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x

def G0(z, c):
    x = Concatenate(axis=1)([c, z])
    x = Dense(128 * 8 * 4 * 4, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Reshape((4, 4, 128 * 8), input_shape=(128 * 8 * 4 * 4,))(x)
    x = upblock(x, 512)
    x = upblock(x, 256)
    x = upblock(x, 128)
    x = upblock(x, 64)
    x = conv3(x, 3)
    x = Activation(activation='tanh')(x)
    return x

def D0(x, c):
    x = Conv2D(64, (4, 4), padding='same', strides=2, input_shape=(64, 64, 3), use_bias=False)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = downblock(x, 128)
    x = downblock(x, 256)
    x = downblock(x, 512)
    c = tile ...
    x = concatenate([x, c])
    x = Conv2D(64 * 8, kernel_size=1, padding='same', strides=1)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Flatten()(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    return x
