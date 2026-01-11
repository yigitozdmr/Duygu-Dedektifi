"""
eğitilmiş RNN modelini kullanarak kullanıcı yorumlarını analiz etme
"""
import numpy as np
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence

#model parametreleri
max_features = 10000  # eğitim modelinde kulalnın max kelime sayisi
maxlen = 200  #rnn modelinin beklediği yorum uzunlugu

#stopwords listesi belirleme
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

#gerekli sozlüklerin oluşturulması: word to index, index to word
word_index = imdb.get_word_index()

#sayilardan kelimelere cevirme sozlugu
index_to_word = {index + 3: word for word, index in word_index.items()}
index_to_word[0] = '<PAD>'
index_to_word[1] = '<START>'
index_to_word[2] = '<UNK>'

#kelimelerden sayilara cevirme sozlugu
word_to_index = {word: index for index, word in index_to_word.items()}

#eiğitim modelini yükleme
model = load_model('rnn_sentiment_model.h5')
print("Model yüklendi.")

#tahmin fonksiyonu
def predict_review(text): #kullanıcıdan gelen metni temizler ve tahmin yapar

    #metni küçük harfe çevirme ve kelimelere ayırma
    words = text_to_word_sequence(text)

    #stopwords temizleme
    cleaned_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]  #sadece harflerden oluşan kelimeler

    encoded_review = [word_to_index.get(word, 2) for word in cleaned_words]  #bilinmeyen kelimeler için 2 kullanılır <UNK>

    #modelin beklediği uzunlukta pad etme
    padded_review = pad_sequences([encoded_review], maxlen=maxlen)

    #tahmin yapma - modelin çıktısı 0-1 arasında bir değer
    prediction = model.predict(padded_review)[0][0]

    print(f"Tahmin sonucu (0-1 arası): {prediction:.4f}")
    if prediction >= 0.5:
        print("Yorum Pozitif.")
    else:
        print("Yorum Negatif.")

#kullanıcıdan yorum alma ve tahmin yapma
user_review = input("Lütfen bir film yorumu giriniz: ")
predict_review(user_review)