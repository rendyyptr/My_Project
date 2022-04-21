#!wget -q -P /content/ https://storage.googleapis.com/tensorflow-1-public/course2/week3/horse-or-human.zip
# Dataset untuk training diatas
# !wget -q -P /content/ https://storage.googleapis.com/tensorflow-1-public/course2/week3/validation-horse-or-human.zip
# Dataset untuk validation
# https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
#     -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
# Tempat download model yang udah di train nanti panggil pake inception
import zipfile
import os
import tensorflow as tf
import matplotlib.pyplot as plt
data_train_ekstrak = zipfile.ZipFile('horse-or-human.zip', 'r')
data_train_ekstrak.extractall('C:/Bangkit_Baru/baru/train')
data_train_ekstrak.close()
data_validation_ekstrak = zipfile.ZipFile('validation-horse-or-human.zip', 'r')
data_validation_ekstrak.extractall('C:/Bangkit_Baru/baru/validation')
data_validation_ekstrak.close()
folder_train = 'C:/Bangkit_Baru/baru/train'
folder_validation = 'C:/Bangkit_Baru/baru/validation'
data_train_horses = os.path.join(folder_train, 'horses')
data_train_human = os.path.join(folder_train, 'humans')
data_test_horses = os.path.join(folder_validation, 'horses')
data_test_humans = os.path.join(folder_validation, 'humans')
# ---------------------------------------------------------------------------------
# Menunjukkan gambar ke-1 untuk data training
plt.imshow(tf.keras.preprocessing.image.load_img(os.listdir(data_train_human)[0]))
plt.show()
# Untuk menunjukkan dimensi gambar diatas
gambar_satu = tf.keras.preprocessing.image.load_img(os.listdir(data_train_human)[0])
gambar_satu_array = tf.keras.preprocessing.image.img_to_array(gambar_satu)
print(gambar_satu_array)
# ----------------------------------------------------------------------------------
train_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0,
                                                                       rotation_range=40,
                                                                       width_shift_range=0.2,
                                                                       height_shift_range=0.2,
                                                                       shear_range=0.2,
                                                                       zoom_range=0.2,
                                                                       horizontal_flip=True,
                                                                       fill_mode='nearest')
train_generator = train_data_generator(directory=folder_train,
                                       batch_size=32,
                                       class_mode='binary',
                                       target_size=(300, 300))
test_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
test_generator = test_data_generator(directory=folder_validation,
                                     batch_size=32,
                                     class_mode='binary',
                                     target_size=(300, 300))
model_savednya = 'C:/Bangkit_Baru/baru/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'


def buat_model_train(model_savednya):
    pre_train = tf.keras.applications.inception_v3.InceptionV3(input_shape=(300, 300, 3),
                                                               include_top=False,
                                                               weight=None)
    for layers in pre_train:
        layers.trainable = False
    return pre_train


pre_train = buat_model_train(model_savednya)
pre_train.summary()
last_layer = pre_train.get_layer('mixed7')
last_output = last_layer.output
# mixed7 bisa diganti apa aja
x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Droput(0.2)(x)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(pre_train.input, x)
model.summary()
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit(train_generator,
                    validation_data = test_generator,
                    steps_per_epoch = 100,
                    epochs = 20,
                    validation_steps = 50,
                    verbose = 2)