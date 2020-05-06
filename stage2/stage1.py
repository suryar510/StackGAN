from tensorflow.keras import Input, Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
import os
import pickle
import random
import time
import PIL
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from PIL import Image
from tensorflow.keras import Input, Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt


def conv3(x, out):
    x = Conv2D(out, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    return x


def upblock(x, out):
    x = UpSampling2D((2, 2), interpolation='nearest')(x)
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
    embed = Input(shape=(1024,))
    x = Dense(256)(embed)
    mulogsigma = LeakyReLU(alpha=0.2)(x)

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
    image = Activation(activation='tanh')(x)

    gen = Model(inputs=[embed, noise], outputs=[image, mulogsigma])
    return gen