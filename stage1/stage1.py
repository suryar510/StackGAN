class Stage1():
    # def __init__(self):
    #     self.ca = self.build_ca()
    #     self.gen = self.build_gen()
    #     self.dis = self.build_dis()

    def conv3(self, x, out):
        x = Conv2D(out, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
        return x

    def upblock(self, x, out):
        x = UpSampling2D((2,2), interpolation='nearest')(x)
        x = conv3(x, out)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

    def downblock(self, x, out):
        x = Conv2D(out, (4, 4), padding='same', strides=2, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    def samplec(self, mulogsigma):
        mu = mulogsigma[:, :128]
        logsigma = mulogsigma[:, 128:]
        stddev = K.exp(logsigma)
        epsilon = K.random_normal(shape=K.constant((mu.shape[1],), dtype='int32'))
        cond = stddev * epsilon + mu
        return cond

    def build_ca(self):
        embed = Input(shape=(1024,))
        x = Dense(256)(embed)
        mulogsigma = LeakyReLU(alpha=0.2)(x)
        model = Model(inputs=[embed], outputs=[mulogsigma])
        return model

    def build_gen(self, ca):
        embed = Input(shape=(1024,))
        mulogsigma = ca(embed)
        cond = Lambda(samplec)(self, mulogsigma)
        noise = Input(shape=(100,))
        x = Concatenate(axis=1)([cond, noise])

        x = Dense(128 * 8 * 4 * 4, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Reshape((4, 4, 128 * 8), input_shape=(128 * 8 * 4 * 4,))(x)

        x = self.upblock(x, 512)
        x = self.upblock(x, 256)
        x = self.upblock(x, 128)
        x = self.upblock(x, 64)

        x = self.conv3(x, 3)
        x = Activation(activation='tanh')(x)

        gen = Model(inputs=[embed, noise], outputs=[x])
        return gen

    def build_dis(self):
        image = Input(shape=(64, 64, 3))

        x = Conv2D(64, (4, 4), padding='same', strides=2, input_shape=(64, 64, 3), use_bias=False)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = self.downblock(x, 128)
        x = self.downblock(x, 256)
        x = self.downblock(x, 512)

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
        embed = Input(shape=(1024,))
        noise = Input(shape=(100,))
        compembed = Input(shape=(4, 4, 128))

        image = gen([embed, noise])

        dis.trainable = False
        score = dis_model([image, compembed])

        gan = Model(inputs=[embed, noise, compembed], outputs=[score])
        return gan
