#!wget --no-check-certificate https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip
import os
import tensorflow as tf
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
data = 'C:/Bangkit_Baru/baru/cats_and_dogs_filtered.zip'
data_ekstrak = zipfile.ZipFile(data, 'r')
data_ekstrak.extractall()
data_ekstrak.close()
folder_name = 'cats_and_dogs_filtered.zip'
data_train = os.path.join(folder_name, 'train')
data_validation = os.path.join(folder_name, 'validation')
data_kucing_train = os.path.join(data_train, 'cats')
data_anjing_train = os.path.join(data_train, 'dogs')
data_kucing_validation = os.path.join(data_validation, 'cats')
data_anjing_validation = os.path.join(data_validation, 'dogs')
# ------------------------------------------------------------------
kolom = 5
baris = 5
indeks_gambar = 0
figure = plt.gcf()
figure.set_size_inches(kolom*5, baris*5)
indeks_gambar += 10
gambar_kucing_train = [os.path.join(data_kucing_train, gambar) for gambar in
                       data_kucing_train[indeks_gambar-10:indeks_gambar]]
gambar_anjing_train = [os.path.join(data_anjing_train, gambar) for gambar in
                       data_anjing_train[indeks_gambar-10:indeks_gambar]]
for i, img_path in enumerate(gambar_kucing_train+gambar_anjing_train):
    size = plt.subplot(baris, kolom, i+1)
    size.axis('off')
    gambar = mpimg.imread(img_path)
    plt.imshow(gambar)
plt.show()
# Diatas ini untuk menunjukan 10 gambar train dari kucing dan anjing
# Sekarang buat modelnya
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(tf.keras.layers.MaxPooling2D(2, 2))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2, 2))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2, 2))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2, 2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
train_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0,
                                                                       rotation_range=40,
                                                                       width_shift_range=0.2,
                                                                       height_shift_range=0.2,
                                                                       shear_range=0.2,
                                                                       zoom_range=0.2,
                                                                       horizontal_flip=True,
                                                                       fill_mode='nearest')
test_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
train_generator = train_data_generator(data_train,
                                       batch_size=10,
                                       class_mode='binary',
                                       target_size=(150, 150))
test_generator = test_data_generator(data_validation,
                                     batch_size=10,
                                     class_mode='binary',
                                     target_size=(150, 150))
model.fit(train_generator,
          step_per_epoch=100,
          epochs=100,
          validation_data=test_generator,
          validation_steps=50,
          verbose=2)
# Train data diatas 100 epoch accuracynya diatas 85%