import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from keras.utils.vis_utils import plot_model

def conv3(x, out):
    x = Conv2D(out, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    return x

def upblock(x, out):
    x = UpSampling2D((2,2), interpolation='nearest')(x)
    x = conv3(x, out)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def downblock(x, out):
    x = Conv2D(out, (4, 4), padding='same', strides=2, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x

def build_ca():
    embed = Input(shape=(1024,))
    x = Dense(256)(embed)
    mulogsigma = LeakyReLU(alpha=0.2)(x)
    ca = Model(inputs=[embed], outputs=[mulogsigma])
    return ca

def build_compembed():
    embed = Input(shape=(1024,))
    x = Dense(128)(embed)
    x = ReLU()(x)
    compembed = Model(inputs=[embed], outputs=[x])
    return compembed

def samplec(mulogsigma):
    mu = mulogsigma[:, :128]
    logsigma = mulogsigma[:, 128:]
    stddev = K.exp(logsigma)
    epsilon = K.random_normal(shape=K.constant((mu.shape[1],), dtype='int32'))
    cond = stddev * epsilon + mu
    return cond

def build_gen():
    inp = Input(shape=(1024,))
    inp_d = Dense(256)(inp)
    mulogsigma = LeakyReLU(alpha=0.2)(inp_d)
    cond = Lambda(samplec)(mulogsigma)
    noise = Input(shape=(100,))
    x = Concatenate(axis=1)([cond, noise])

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

    gen = Model(inputs=[inp, noise], outputs=[x, mulogsigma])
    return gen

def build_dis():
    image = Input(shape=(64, 64, 3))
    x = Conv2D(64, (4, 4), padding='same', strides=2, input_shape=(64, 64, 3), use_bias=False)(image)
    x = LeakyReLU(alpha=0.2)(x)
    x = downblock(x, 128)
    x = downblock(x, 256)
    x = downblock(x, 512)

    compembed = Input(shape=(4, 4, 128))
    x = concatenate([x, compembed])
    x = Conv2D(64 * 8, kernel_size=1, padding='same', strides=1)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Flatten()(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)

    dis = Model(inputs=[image, compembed], outputs=[x])
    return dis

def build_gan(gen, dis):
    mulogsigma = Input(shape=(1024,))
    noise = Input(shape=(100,))
    compembed = Input(shape=(4, 4, 128))

    image, m = gen([mulogsigma, noise])

    dis.trainable = False
    score = dis([image, compembed])

    gan = Model(inputs=[mulogsigma, noise, compembed], outputs=[score, m])
    return gan

ca = build_ca()
compembed = build_compembed()
gen = build_gen()
dis = build_dis()
gan = build_gan(gen, dis)
#print(ca.summary())
#print(compembed.summary())
print(gen.summary())
#print(dis.summary())
#print(gan.summary())
print("===============================================================================")
# plot_model(ca, to_file='ca.png', show_shapes=True, show_layer_names=True)
# plot_model(compembed, to_file='compembed.png', show_shapes=True, show_layer_names=True)
# plot_model(gen, to_file='gen.png', show_shapes=True, show_layer_names=True)
# plot_model(dis, to_file='dis.png', show_shapes=True, show_layer_names=True)
# plot_model(gan, to_file='gan.png', show_shapes=True, show_layer_names=True)


def generate_c(x):
    mean = x[:, :128]
    log_sigma = x[:, 128:]
    stddev = K.exp(log_sigma)
    epsilon = K.random_normal(shape=K.constant((mean.shape[1],), dtype='int32'))
    c = stddev * epsilon + mean
    return c

def build_ca_model():
    """
    Get conditioning augmentation model.
    Takes an embedding of shape (1024,) and returns a tensor of shape (256,)
    """
    input_layer = Input(shape=(1024,))
    x = Dense(256)(input_layer)
    x = LeakyReLU(alpha=0.2)(x)
    model = Model(inputs=[input_layer], outputs=[x])
    return model

def build_embedding_compressor_model():
    """
    Build embedding compressor model
    """
    input_layer = Input(shape=(1024,))
    x = Dense(128)(input_layer)
    x = ReLU()(x)

    model = Model(inputs=[input_layer], outputs=[x])
    return model

def build_stage1_generator():
    """
    Builds a generator model used in Stage-I
    """
    input_layer = Input(shape=(1024,))
    x = Dense(256)(input_layer)
    mean_logsigma = LeakyReLU(alpha=0.2)(x)

    c = Lambda(generate_c)(mean_logsigma)

    input_layer2 = Input(shape=(100,))

    gen_input = Concatenate(axis=1)([c, input_layer2])

    x = Dense(128 * 8 * 4 * 4, use_bias=False)(gen_input)
    x = ReLU()(x)

    x = Reshape((4, 4, 128 * 8), input_shape=(128 * 8 * 4 * 4,))(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(512, kernel_size=3, padding="same", strides=1, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(256, kernel_size=3, padding="same", strides=1, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, kernel_size=3, padding="same", strides=1, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, kernel_size=3, padding="same", strides=1, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(3, kernel_size=3, padding="same", strides=1, use_bias=False)(x)
    x = Activation(activation='tanh')(x)

    stage1_gen = Model(inputs=[input_layer, input_layer2], outputs=[x, mean_logsigma])
    return stage1_gen

def build_stage1_discriminator():
    """
    Create a model which takes two inputs
    1. One from above network
    2. One from the embedding layer
    3. Concatenate along the axis dimension and feed it to the last module which produces final logits
    """
    input_layer = Input(shape=(64, 64, 3))

    x = Conv2D(64, (4, 4),
               padding='same', strides=2,
               input_shape=(64, 64, 3), use_bias=False)(input_layer)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(128, (4, 4), padding='same', strides=2, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(256, (4, 4), padding='same', strides=2, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(512, (4, 4), padding='same', strides=2, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    input_layer2 = Input(shape=(4, 4, 128))

    merged_input = concatenate([x, input_layer2])

    x2 = Conv2D(64 * 8, kernel_size=1,
                padding="same", strides=1)(merged_input)
    x2 = BatchNormalization()(x2)
    x2 = LeakyReLU(alpha=0.2)(x2)
    x2 = Flatten()(x2)
    x2 = Dense(1)(x2)
    x2 = Activation('sigmoid')(x2)

    stage1_dis = Model(inputs=[input_layer, input_layer2], outputs=[x2])
    return stage1_dis

def build_adversarial_model(gen_model, dis_model):
    input_layer = Input(shape=(1024,))
    input_layer2 = Input(shape=(100,))
    input_layer3 = Input(shape=(4, 4, 128))

    x, mean_logsigma = gen_model([input_layer, input_layer2])

    dis_model.trainable = False
    valid = dis_model([x, input_layer3])

    model = Model(inputs=[input_layer, input_layer2, input_layer3], outputs=[valid, mean_logsigma])
    return model

ca1 = build_ca_model()
compembed1 = build_embedding_compressor_model()
gen1 = build_stage1_generator()
dis1 = build_stage1_discriminator()
gan1 = build_adversarial_model(gen1, dis1)
#print(ca1.summary())
#print(compembed1.summary())
print(gen1.summary())
#print(dis1.summary())
#print(gan1.summary())
