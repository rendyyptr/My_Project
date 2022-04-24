import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
tokenizer = info.features['text'].encoder
train_data, test_data = dataset['train'], dataset['test']
train_dataset = train_data.shuffle(10000)
train_dataset = train_dataset.padded_batch(256)
test_dataset = test_data.padded_batch(256)
# --------------------------------------------------------------------------------------
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(tokenizer.vocab_size, 64))
model.add(tf.keras.layers.Conv1D(128, kernel_size=5, activation='relu'))
model.add(tf.keras.layers.GlobalMaxPooling1D())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
# --------------------------------------------------------------------------------------
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_dataset, epochs=10, validation_data=test_dataset)
# --------------------------------------------------------------------------------------


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()


plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')
