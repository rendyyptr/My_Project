import csv
from tensorflow import keras
from keras import preprocessing
with open('C:/Bangkit_Baru/baru/CAPSTONE_LATIHAN/bbc-text.csv') as csvfile:
    print(f'Ini Header : \n\n {csvfile.readline()}')
    print(f'Data nya bernilai seperti ini : \n\n{csvfile.readline()}')


def hapus_kata_terlarang(sentence):
    terlarang = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
                 "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did",
                 "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
                 "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself",
                 "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's",
                 "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only",
                 "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll",
                 "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
                 "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
                 "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll",
                 "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while",
                 "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've",
                 "or", "your", "yours", "yourself", "yourselves"]
    sentence = sentence.lower().split()
    sentence = [kata for kata in sentence if kata not in terlarang]
    sentence = ' '.join(sentence)
    return sentence


# Tes dulu fungsinya ges
print(hapus_kata_terlarang('I am about to go to the store and get any snack'))
# Parse data cari sentence dan labelnya


def parse_data(filename):
    sentences = []
    labels = []
    with open(filename, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            labels.append(row[0])
            sentence = row[1]
            new_sentence = hapus_kata_terlarang(sentence)
            sentences.append(new_sentence)
        del labels[0]
        del sentences[0]
    return sentences, labels


sentences, labels = parse_data('C:/Bangkit_Baru/baru/CAPSTONE_LATIHAN/bbc-text.csv')
print(f'Jumlah sentence yang ada : {len(sentences)}')
print(f'Jumlah sentence setelah dihapus kata kata terlarangnya : {len(sentences[0].split())}')
print(f'Jumlah label yang ada : {len(labels)}')
print(f'5 label diawal : {labels[:5]}')


def buat_tokenizer_sekalian_pad(sentence):
    tokenizer = preprocessing.text.Tokenizer(oov_token='<OOV>')
    tokenizer.fit_on_texts(sentence)
    # print(f'{tokenizer}\n\n, ini adalah word indexnya {tokenizer.word_index}')
    print(f'jumlah word index : {len(tokenizer.word_index)}')
    print('<OOV> ada di dalamnya' if '<OOV>' in tokenizer.word_index else '<OOV> gaada didalamnya')
    sequences = tokenizer.texts_to_sequences(sentence)
    padded_sequences = preprocessing.sequence.pad_sequences(sequences, padding='post')
    print(f'ini adalah padded sequence pertama : {padded_sequences[0]}')
    print(f'ini bentuknya : {padded_sequences.shape}')


buat_tokenizer_sekalian_pad(sentences)


def tokenisasi_labels(labels):
    label_tokenizer = preprocessing.text.Tokenizer()
    label_tokenizer.fit_on_texts(labels)
    label_word_index = label_tokenizer.word_index
    label_sequence = label_tokenizer.texts_to_sequences(labels)
    return label_sequence, label_word_index


label_sequence, label_word_index = tokenisasi_labels(labels)
print(f'ini word index : {label_word_index}')
print(f'ini label sequence : {label_sequence}')