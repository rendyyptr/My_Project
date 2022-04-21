from tensorflow import keras
from keras import preprocessing
sentences = [
    'i love my dog',
    'I, love my cat',
    'You love my dog!',
    'Do you think my dog stink?'
]
tokenizer = preprocessing.text.Tokenizer(num_words=100, oov_token='<OOV>')
# Generate indices for each word in the corpus
tokenizer.fit_on_texts(sentences)
# Get the indices and print and put it to corpus
word_index = tokenizer.word_index
# Generate list of index and list it
sequences = tokenizer.texts_to_sequences(sentences)
print('\nWord Index', word_index)
print('\nSequences', sequences)
# PADDING
padded = preprocessing.sequence.pad_sequences(sequences, maxlen=5) # 5 = n
print('\nPadded Sequences')
print(padded)
# padding membuat ia menjadi satu matriks dengan jumlah dimensi n x n
test_data = [
    'I love my horse',
    'Do you think im stink?'
]
test_seq = tokenizer.texts_to_sequences(test_data)
padded = preprocessing.sequence.pad_sequences(test_seq, maxlen=10)
print('\nPadded test sequence')
print(padded)
# OOV = Out of Vocabulary memberikan angka pada kalimat yang tidak ada di corpus
