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

    def samplec(mulogsigma):
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
        ca = Model(inputs=[embed], outputs=[mulogsigma])
        return ca

    def build_compembed(self):
        embed = Input(shape=(1024,))
        x = Dense(128)(embed)
        x = ReLU()(x)
        compembed = Model(inputs=[embed], outputs=[x])
        return compembed

    def build_gen(self, ca):
        embed = Input(shape=(1024,))
        mulogsigma = ca(embed)
        cond = Lambda(samplec)(mulogsigma)
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

    def build_gan(self, gen, dis):
        embed = Input(shape=(1024,))
        noise = Input(shape=(100,))
        compembed = Input(shape=(4, 4, 128))

        image = gen([embed, noise])

        dis.trainable = False
        score = dis([image, compembed])

        gan = Model(inputs=[embed, noise, compembed], outputs=[score])
        return gan

    def train(self):
        self.data_dir = "/content/birds/"
        self.train_dir = self.data_dir + "/train"
        self.test_dir = self.data_dir + "/test"
        self.image_size = 64
        self.batch_size = 64
        self.noise_dim = 100
        self.gen_lr = 0.0002
        self.dis_lr = 0.0002
        self.lr_decay_step = 600
        self.epochs = 1000
        self.cond_dim = 128

        embeddings_file_path_train = self.train_dir + "/char-CNN-RNN-embeddings.pickle"
        embeddings_file_path_test = self.test_dir + "/char-CNN-RNN-embeddings.pickle"

        filenames_file_path_train = self.train_dir + "/filenames.pickle"
        filenames_file_path_test = self.test_dir + "/filenames.pickle"

        class_info_file_path_train = self.train_dir + "/class_info.pickle"
        class_info_file_path_test = self.test_dir + "/class_info.pickle"

        cub_dataset_dir = "/content/CUB_200_2011"

        X_train, y_train, embed_train = load_dataset(filenames_file_path=filenames_file_path_train,
                                                          class_info_file_path=class_info_file_path_train,
                                                          cub_dataset_dir=cub_dataset_dir,
                                                          embeddings_file_path=embeddings_file_path_train,
                                                          self.image_size=(64, 64))

        X_test, y_test, embed_test = load_dataset(filenames_file_path=filenames_file_path_test,
                                                       class_info_file_path=class_info_file_path_test,
                                                       cub_dataset_dir=cub_dataset_dir,
                                                       embeddings_file_path=embeddings_file_path_test,
                                                       self.image_size=(64, 64))

#===================================================================================================================

        self.ca = self.build_ca()
        self.ca.compile(loss="binary_crossentropy", optimizer="adam")

        self.dis = self.build_dis()
        dis_optimizer = Adam(lr=self.dis_lr, beta_1=0.5, beta_2=0.999)
        self.dis.compile(loss='binary_crossentropy', optimizer=dis_optimizer)

        self.gen = self.build_gen(self.ca)
        gen_optimizer = Adam(lr=self.gen_lr, beta_1=0.5, beta_2=0.999)
        self.gen.compile(loss="mse", optimizer=gen_optimizer)

        self.compembed = self.build_compembed()
        self.compembed.compile(loss="binary_crossentropy", optimizer="adam")

        def KL_loss(y_true, y_pred):
            mu = y_pred[:, :128]
            logsigma = y_pred[:, :128]
            loss = -logsigma + .5 * (-1 + K.exp(2. * logsigma) + K.square(mu))
            loss = K.mean(loss)
            return loss

        def custom_generator_loss(y_true, y_pred):
            return K.binary_crossentropy(y_true, y_pred)

        self.gan = self.build_gan(self.gen, self.dis)
        self.gan.compile(loss=['binary_crossentropy', KL_loss], loss_weights=[1, 2.0],
                                  optimizer=gen_optimizer, metrics=None)

        self.tensorboard = TensorBoard(log_dir="logs/".format(time.time()))
        self.tensorboard.set_model(self.gen)
        self.tensorboard.set_model(self.dis)
        self.tensorboard.set_model(self.ca)
        self.tensorboard.set_model(self.compembed)

        # Generate an array containing real and fake values
        # Apply label smoothing as well
        real_labels = np.ones((self.batch_size, 1), dtype=float) * 0.9
        fake_labels = np.zeros((self.batch_size, 1), dtype=float) * 0.1

        for epoch in range(self.epochs):
            print("========================================")
            print("Epoch:", epoch)
            print("Number of batches:", int(X_train.shape[0] / self.batch_size))

            gen_losses = []
            dis_losses = []

            # Load data and train model
            num_batches = int(X_train.shape[0] / self.batch_size)
            for index in range(num_batches):
                print("Batch:{}".format(index+1))

                # Sample a batch of data
                noise_batch = np.random.normal(0, 1, size=(self.batch_size, self.noise_dim))
                image_batch = X_train[index * self.batch_size:(index + 1) * self.batch_size]
                image_batch = (image_batch - 127.5) / 127.5
                embed_batch = embed_train[index * self.batch_size:(index + 1) * self.batch_size]

                # Generate fake images
                fake_images = self.gen.predict([embed_batch, noise_batch], verbose=3)

                # Generate compressed embeddings
                compembed_batch = self.compembed.predict_on_batch(embed_batch)
                compembed_batch = np.reshape(compembed_batch, (-1, 1, 1, self.cond_dim))
                compembed_batch = np.tile(compembed_batch, (1, 4, 4, 1))

                dis_loss_real = self.dis.train_on_batch([image_batch, compembed_batch],
                                                          np.reshape(real_labels, (self.batch_size, 1)))
                dis_loss_fake = self.dis.train_on_batch([fake_images, compembed_batch],
                                                          np.reshape(fake_labels, (self.batch_size, 1)))
                dis_loss_wrong = self.dis.train_on_batch([image_batch[:(self.batch_size - 1)], compembed_batch[1:]],
                                                           np.reshape(fake_labels[1:], (self.batch_size-1, 1)))
                dis_loss = 0.5 * np.add(dis_loss_real, 0.5 * np.add(dis_loss_wrong, dis_loss_fake))
                print("d_loss_real:{}".format(dis_loss_real))
                print("d_loss_fake:{}".format(dis_loss_fake))
                print("d_loss_wrong:{}".format(dis_loss_wrong))
                print("dis_loss:{}".format(dis_loss))

                gen_loss = self.gan.train_on_batch([embed_batch, noise_batch, compembed_batch],[K.ones((self.batch_size, 1)) * 0.9, K.ones((self.batch_size, 256)) * 0.9])
                print("gen_loss:{}".format(gen_loss))

                dis_losses.append(dis_loss)
                gen_losses.append(gen_loss)

            self.write_log(self.tensorboard, 'dis_loss', np.mean(dis_losses), epoch)
            self.write_log(self.tensorboard, 'gen_loss', np.mean(gen_losses[0]), epoch)

            # Generate and save images after every 2nd epoch
            if epoch % 2 == 0:
                # z_noise2 = np.random.uniform(-1, 1, size=(self.batch_size, self.noise_dim))
                noise2 = np.random.normal(0, 1, size=(self.batch_size, self.noise_dim))
                embed_batch = embed_test[0:self.batch_size]
                fake_images = self.gen.predict_on_batch([embed_batch, noise2])

                # Save images
                for i, img in enumerate(fake_images[:10]):
                    self.save_rgb_img(img, "results/gen_{}_{}.png".format(epoch, i))

        # Save models
        self.gen.save_weights("stage1_gen.h5")
        self.dis.save_weights("stage1_dis.h5")

    def save_rgb_img(self, image, path):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(image)
        ax.axis("off")
        ax.set_title("Image")
        plt.savefig(path)
        plt.close()

    def write_log(callback, name, loss, batch_no):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = loss
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()
