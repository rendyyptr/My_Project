import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
data = open('C:/Bangkit_Baru/baru/CAPSTONE_LATIHAN/irish-lyrics-eof.txt').read()
# split dan buat lower
corpus = data.lower().split()
# print(corpus)
tokenizer = Tokenizer()
# Buat dictiionarynya
tokenizer.fit_on_texts(corpus)
# Total kata di dictionary
total_words = len(tokenizer.word_index)+1
# print(tokenizer.word_index)
# print(total_words)
input_sequence = []
for words in corpus:
    # buat listnya
    list_token = tokenizer.texts_to_sequences([words])[0]
    for i in range(1, len(list_token)):
        # generate subphrase
        n_sequence = list_token[:i+1]
        input_sequence.append(n_sequence)
# Menampilkan kalimat yang paling panjang
max_sequence = max([len(x) for x in input_sequence])
# padding
input_sequence = np.array(pad_sequences(input_sequence, maxlen=max_sequence, padding='pre'))
# Buat input dan labels
input, labels = input_sequence[:,:-1]
#Konversi ke one hot array
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)
# ---------------------------------------------------------------------------------------------------
sentence = corpus[2].split()
list_word_index = []
for word in sentence:
    list_word_index.append(tokenizer.word_index[word])
print(list_word_index)
# ---------------------------------------------------------------------------------------------------
elem_number = 19
print(f'token list : {input[elem_number]}')
print(f'decoded to text : {tokenizer.sequences_to_texts(input[elem_number])}')
# ---------------------------------------------------------------------------------------------------
# ini label
print(f'token list : {ys[elem_number]}')
print(f'token list : {np.argmax(ys[elem_number])}')
# ---------------------------------------------------------------------------------------------------
model = tf.keras.models.Sequential()
model,add(tf.keras.layers.Embedding(total_words, 64, input_length=max_sequence-1))
model.add(tf.keras.layers.Bidirectional(LSTM(20)))
model.add(tf.keras.layers.Dense(total_words, activation='softmax'))
# ---------------------------------------------------------------------------------------------------
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(input, ys, epoch=500)
def buat_grafik(history, string):
    plt.plot(history.history[string])
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.show
buat_grafik((history, 'accuracy'))
teks = 'Always beware and keep your garden'
for i in range(num):
    # Convert ke token sequence
    token_list = tokenizer.texts_to_sequences([teks])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence-1, padding='pre')
    probabilitas = model.predict(token_list)
    predicted = np.argmax(probabilitas, axis=-1)[0]
    if predicted != 0:
        hasil = tokenizer.word_index([predicted])
        teks = ' ' + hasil
print(teks)
