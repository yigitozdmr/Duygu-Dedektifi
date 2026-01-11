"""
RNN ile Duygu Dedektifi (Sentiment Analysis)

Problem Tanimi: Bir yorumun olumlu mu yoksa olumsuz mu olduğunu belirlemek.
    IMDB film yorumlari veri setini kullanarak bir metnin duygusal analizlerini gerçekleştirmek.
    - this movie is awesome -> positive
    - this movie is terrible -> negative

RNN:Tekrarlayan sinir ağı (RNN), zaman serisi veya sıralı veriler üzerinde tahmin yapan derin öğrenme için bir ağ mimarisidir.

Girdi: film -> cok -> kotuydu
Belllek:
Cikti: anlam anlam olumsuz

Veri seti: IMDB film yorumlari veri seti (olumlu ve olumsuz)

plan/program

Gerekli kurulumlar:

import libraries
*terminal sanal ortam açma: .\venv\Scripts\activate
"""

#import libraries
import numpy as np
import nltk 
import matplotlib.pyplot as plt
from nltk.corpus import stopwords #Gereksiz kelime listesi
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers #Modelin ağırlıklarının çok büyümesini (ezberlemesini) engellemek için

#stopwords listesi belirleme
nltk.download('stopwords') #nltk içinden ingilizce stopwords indiriyor
stop_words = set(stopwords.words('english'))

#model parametreleri
max_features = 10000  #kelime sayisi

#load dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

#ornek veri inceleme  
original_word_index = imdb.get_word_index()

#sayi kelime cevirme 
inv_word_index = {index + 3: word for word, index in original_word_index.items()}
inv_word_index[0] = '<PAD>' #0: padding-boşluk
inv_word_index[1] = '<START>' #1: başlangıç
inv_word_index[2] = '<UNK>' #2: bilinmeyen kelime

#sayi dizisini kelime dizisine çevirme fonksiyonu
def decode_review(text):
    return ' '.join([inv_word_index.get(i, '?') for i in text])

movie_index = 0
#ilk eğitim verisini yazdıralım
print("ilk yorum: (sayi dizisi)")
print(X_train[movie_index])

print("ilk yorum: (kelimelerle)")
print(decode_review(X_train[movie_index]))

print(f"label: {'Pozitif' if y_train[movie_index]==1 else 'Negatif'}")

#gerekli sozlüklerin oluşturulması: word to index, index to word
word_index = imdb.get_word_index()
index_to_word = {index + 3: word for word, index in word_index.items()}
index_to_word[0] = '<PAD>'
index_to_word[1] = '<START>'
index_to_word[2] = '<UNK>'
word_to_index = {word: index for index, word in index_to_word.items()} #kelimelerden sayilara geçiş

#data preprocessing(veri ön işleme)
def preprocess_review(encoded_review):
    #sayiları kelimelere çevirme
    words = [index_to_word.get(i, "") for i in encoded_review if i >= 3]

    #sadece harflerden oluşan kelimeleri alma ve stopwords kaldırma
    cleaned = [
        word.lower() for word in words 
        if word.isalpha() and word.lower() not in stop_words
    ]

    #tekrardan temizlenmiş kelimeleri sayılara çevirme
    return [word_to_index.get(word, 2) for word in cleaned] #bilinmeyen kelimeler için 2 kullanılır

#veriyi temizle ve sabit uzunlugu pad et
X_train = [preprocess_review(review) for review in X_train]
X_test = [preprocess_review(review) for review in X_test]

#pad sequence
X_train = pad_sequences(X_train, maxlen=200)
X_test = pad_sequences(X_test, maxlen=200)

#RNN Modeli oluşturma
model = Sequential() # base model: katmanları sıralı şekilde ekleme için

#embedding katmanı: kelime gömme
model.add(Embedding(input_dim=max_features, output_dim=16))

#1. LSTM katmanı: sıralı çıktıyı bir sonraki LSTM'e vermek için
model.add(
    LSTM(
        8,
        return_sequences=True,
        dropout=0.3,
        recurrent_dropout=0.3,
        kernel_regularizer=regularizers.l2(0.001),
        recurrent_regularizer=regularizers.l2(0.001)
    )
)

#YENİ: Dropout katmanı ekleme (aşırı öğrenmeyi önlemek için)
model.add(Dropout(0.5))

#2. LSTM katmanı: son özet temsili çıkarır
model.add(
    LSTM(
        8,
        dropout=0.3,
        recurrent_dropout=0.3,
        kernel_regularizer=regularizers.l2(0.001),
        recurrent_regularizer=regularizers.l2(0.001)
    )
)

#output katmanı
model.add(Dense(1, activation='sigmoid'))

#model compile etme
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

#training
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop]
)

#model evaluation plot
def plot_history(hist):
    plt.figure(figsize=(12, 4))

    #accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(hist.history['accuracy'], label='Train Accuracy')
    plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    #loss plot
    plt.subplot(1, 2, 2)
    plt.plot(hist.history['loss'], label='Train Loss')
    plt.plot(hist.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_history(history)

#test verisi ile model değerlendirme
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")

#eğitilen verinin kaydedilmesi
model.save("rnn_sentiment_model.h5")
print("Model kaydedildi: rnn_sentiment_model.h5")

