import json
from tensorflow import keras
from keras import preprocessing
with open('C:/Users/lenovo PC/Downloads/tensorflow-2-public-main/sarcasm.json', 'r') as f:
    data = json.load(f)
print(data[0])
print(data[2000])
isi = []
labels = []
link = []
for item in data:
    isi.append(item['headline'])
    labels.append(item['is_sarcastic'])
    link.append(item['article_link'])
tokenizer = preprocessing.text.Tokenizer(oov_token='OOV')
tokenizer.fit_on_texts(isi)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(isi)
padded = preprocessing.sequence.pad_sequences(sequences, padding='post')
print(f'Sample Headline: {isi[2]}]')
print(f'Padded Sequences: {padded[2]}')
print(padded.shape)