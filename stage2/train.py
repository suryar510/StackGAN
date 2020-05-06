from keras import Model, Input
from keras.optimizers import Adam
from keras.backend import mean, exp, square, ones
import stage2
import stage1
import numpy as np
import pickle
import os
import pandas as pd
import PIL
from PIL import Image
import random
import matplotlib as plt


def load(classes, embeddings, filenames):
    with open(classes, 'rb') as f:
        class_ids = pickle.load(f, encoding='latin1')

    with open(embeddings, 'rb') as f:
        embeddings = pickle.load(f, encoding='latin1')
        embeddings = np.array(embeddings)

    with open(filenames, 'rb') as f:
        filenames = pickle.load(f, encoding='latin1')

    return class_ids, embeddings, filenames


def load_bounding_boxes(dataset_dir):
    bounding_boxes_path = os.path.join(dataset_dir, 'bounding_boxes.txt')
    file_paths_path = os.path.join(dataset_dir, 'images.txt')

    df_bounding_boxes = pd.read_csv(bounding_boxes_path,
                                    delim_whitespace=True, header=None).astype(int)
    df_file_names = pd.read_csv(file_paths_path, delim_whitespace=True, header=None)

    # Create a list of file names
    file_names = df_file_names[1].tolist()

    # Create a dictionary of file_names and bounding boxes
    filename_boundingbox_dict = {img_file[:-4]: [] for img_file in file_names[:2]}

    # Assign a bounding box to the corresponding image
    for i in range(0, len(file_names)):
        # Get the bounding box
        bounding_box = df_bounding_boxes.iloc[i][1:].tolist()
        key = file_names[i][:-4]
        filename_boundingbox_dict[key] = bounding_box

    return filename_boundingbox_dict


def get_img(img_path, bbox, image_size):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        R = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - R)
        y2 = np.minimum(height, center_y + R)
        x1 = np.maximum(0, center_x - R)
        x2 = np.minimum(width, center_x + R)
        img = img.crop([x1, y1, x2, y2])
    img = img.resize(image_size, PIL.Image.BILINEAR)
    return img


def load_dataset(files, classes, embeds, cub, size):
    class_ids, filed_embeddings, filenames = load(classes, embeds, files)
    bounding_boxes = load_bounding_boxes(cub)
    X, y, embeddings = [], [], []

    for index, filename in enumerate(filenames):
        box = bounding_boxes[filename]

        try:
            # Load images
            name = cub + '/images/' + filename + '.jpg'
            image = get_img(name, box, size)

            all_embeddings = filed_embeddings[index, :, :]
            embedding_ix = random.randint(0, all_embeddings.shape[0] - 1)
            embedding = all_embeddings[embedding_ix, :]

            X.append(np.array(image))
            y.append(class_ids[index])
            embeddings.append(embedding)
        except Exception as e:
            print(e)
    X_data = np.asarray(X, dtype=np.float32)
    y_data = np.asarray(y, dtype=np.float32)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    return X_data, y_data, embeddings



def buildModel(s1gen, s2gen, s2dis):
    s1gen.trainable = False
    s2dis.trainable = False

    embed_inp = Input(shape=(1024,))
    noise = Input(shape=(100,))
    compressed_embed = Input(shape=(4, 4, 128))

    s1_img, _ = s1gen([embed_inp, noise])
    s2_img, text = s2gen([embed_inp, s1_img])
    out = s2_dis([s2_img, compressed_embed])

    mod = Model(inputs=[embed_inp, noise, compressed_embed], outputs=[out, text])
    return mod

def klloss(true, pred):
    mu = pred[:, :128]
    logsigma = pred[:, :128]
    loss = -logsigma + .5 * (-1 + exp(2. * logsigma) + square(mu))
    loss = mean(loss)
    return loss

def save_rgb_img(image, path):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(image)
    ax.axis("off")
    ax.set_title("Image")
    plt.savefig(path)
    plt.close()

if __name__ == '__main__':

    data_dir = "/Users/suryar/Documents/ucb/mlab/stackgan/birds"
    train_dir = data_dir + "/train"
    test_dir = data_dir + "/test"
    batch_size = 64
    noise_dim = 100
    lr_decay_step = 600
    epochs = 1000
    cond_dim = 128

    embeddings_file_path_train = train_dir + "/char-CNN-RNN-embeddings.pickle"
    embeddings_file_path_test = test_dir + "/char-CNN-RNN-embeddings.pickle"

    filenames_file_path_train = train_dir + "/filenames.pickle"
    filenames_file_path_test = test_dir + "/filenames.pickle"

    class_info_file_path_train = train_dir + "/class_info.pickle"
    class_info_file_path_test = test_dir + "/class_info.pickle"

    cub_dataset_dir = data_dir+"/CUB_200_2011"

    print("Loading Training")
    X_train, y_train, embed_train = load_dataset(files=filenames_file_path_train,
                                                 classes=class_info_file_path_train,
                                                 embeds=embeddings_file_path_train,
                                                 cub=cub_dataset_dir,
                                                 size=(256, 256))
    print("Loading Testing")
    X_test, y_test, embed_test = load_dataset(files=filenames_file_path_train,
                                              classes=class_info_file_path_train,
                                              embeds=embeddings_file_path_train,
                                              cub=cub_dataset_dir,
                                              size=(256, 256))

    s2_gen = stage2.generator()
    gen_opt = Adam(lr=.0002, beta_1=.5, beta_2=.99)
    s2_gen.compile(optimizer=gen_opt, loss='binary_crossentropy', metrics=['accuracy'])

    s1_gen = stage1.build_gen()
    s1_gen.compile(optimizer=gen_opt, loss="binary_crossentropy")
    # s1_gen.load_weights("s1_gen.h5")

    s2_dis = stage2.discriminator()
    dis_opt = Adam(lr=.0002, beta_1=.5, beta_2=.99)
    s2_dis.compile(optimizer=dis_opt, loss='binary_crossentropy', metrics=['accuracy'])

    compembed = stage1.build_compembed()
    compembed.compile(loss="binary_crossentropy", optimizer="adam")

    model = buildModel(s1_gen, s2_gen, s2_dis)
    model.compile(optimizer=gen_opt, loss=['binary_crossentropy', klloss], loss_weights=[1.0, 2.0], metrics=None)

    real_labels = np.ones((batch_size, 1), dtype=float) * 0.9
    fake_labels = np.zeros((batch_size, 1), dtype=float) * 0.1

    for epoch in range(epochs):
        print("=====================\nEPOCH: "+epoch)

        gen_losses = []
        dis_losses = []

        num_batch = len(X_train.shape[0]) / batch_size
        for index in range(num_batch):
            print("\tBatch: " + index)
            # pass through generator model
            noise_batch = np.random.normal(0, 1, size=(batch_size, noise_dim))
            image_batch = X_train[index * batch_size:(index + 1) * batch_size]
            image_batch = (image_batch - 127.5) / 127.5
            embed_batch = embed_train[index * batch_size:(index + 1) * batch_size]

            fake_image_1, _ = s1_gen.predict([embed_batch, noise_batch], verbose=3)
            fake_image_2, _ = s2_gen.predict([embed_batch, fake_image_1], verbose=3)

            compembed_batch = compembed.predict_on_batch(embed_batch)
            compembed_batch = np.reshape(compembed_batch, (-1, 1, 1, cond_dim))
            compembed_batch = np.tile(compembed_batch, (1, 4, 4, 1))

            dis_loss_real = s2_dis.train_on_batch([image_batch, compembed_batch],
                                               np.reshape(real_labels, (batch_size, 1)))
            dis_loss_fake = s2_dis.train_on_batch([fake_image_2, compembed_batch],
                                               np.reshape(fake_labels, (batch_size, 1)))
            dis_loss_wrong = s2_dis.train_on_batch([image_batch[:(batch_size - 1)], compembed_batch[1:]],
                                                np.reshape(fake_labels[1:], (batch_size - 1, 1)))
            dis_loss = 0.5 * np.add(dis_loss_real, 0.5 * np.add(dis_loss_wrong, dis_loss_fake))
            print("\t\td_loss_real:{}".format(dis_loss_real))
            print("\t\td_loss_fake:{}".format(dis_loss_fake))
            print("\t\td_loss_wrong:{}".format(dis_loss_wrong))
            print("\t\tdis_loss:{}".format(dis_loss))

            loss = model.train_on_batch([embed_batch, noise_batch, compembed_batch], [ones((batch_size, 1)) * 0.9, ones((batch_size, 256)) * 0.9])
            print("\t\tmodel_loss:{}".format(loss))
            dis_losses.append(dis_loss)
            gen_losses.append(loss)
            # train discriminator on fake, real, and wrong images
            # train adversarial model
        if epoch % 2 == 0:
            print("Saving weights")
            s2_dis.save_weights("s2_dis_%d.h5" % epoch)
            s2_gen.save_weights("s2_gen_%d.h5" % epoch)

            test_noise = np.random.normal(0, 1, size=(batch_size, noise_dim))
            test_embed = embed_test[0:batch_size]
            s1_img, _ = s1_gen.predict([test_embed, test_noise], verbose=3)
            s2_img, _ = s2_gen.predict([test_embed, s1_img], verbose=3)

            for idx, img in enumerate(s2_img[:5]):
                save_rgb_img(img, "logs/img{}_{}.png".format(epoch, idx))

        if epoch % 100 == 0:
            gen_opt.learning_rate /= 2
            dis_opt.learning_rate /= 2

