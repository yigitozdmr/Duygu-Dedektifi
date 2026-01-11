Duygu Dedektifi (Sentiment Analysis with RNN)

Bu proje, IMDB film yorumlarını analiz ederek yorumun **Olumlu (Positive)** mu yoksa **Olumsuz (Negative)** mu olduğunu tahmin eden bir Yapay Zeka uygulamasıdır. Derin Öğrenme yöntemlerinden **RNN (Recurrent Neural Networks)** ve **LSTM** mimarisi kullanılarak geliştirilmiştir.

## 🚀 Özellikler

- **Veri Seti:** IMDB Film Yorumları (25.000 Eğitim - 25.000 Test)
- **Model Mimarisi:** Embedding Layer + LSTM + Dropout + Dense Layer
- **Teknoloji:** TensorFlow / Keras, Python, NLTK
- **Başarım:** Model, test verileri üzerinde yüksek doğruluk oranı ile duygu analizi yapabilmektedir.

## 📂 Kurulum

Projeyi bilgisayarınıza indirdikten sonra gerekli kütüphaneleri yüklemek için:

```bash
pip install -r requirements.txt
⚙️ Kullanım
1. Modeli Eğitmek
Eğer modeli sıfırdan eğitmek isterseniz:

Bash

python train_rnn_model.py
Bu işlem sonucunda rnn_sentiment_model.keras dosyası oluşturulacaktır.

2. Tahmin Yapmak
Eğitilmiş modeli kullanarak kendi cümlenizi test etmek için:

Bash

python predict_rnn_model.py
Program sizden bir İngilizce film yorumu girmenizi isteyecektir.

Örnek:

Lütfen bir film yorumu giriniz: The movie was fantastic and acting was great! Sonuç: Yorum Pozitif.

🧠 Model Mimarisi Hakkında
Projede kelime sırasını ve bağlamı yakalayabilmek için LSTM (Long Short-Term Memory) hücreleri kullanılmıştır. Aşırı öğrenmeyi (Overfitting) engellemek için Dropout ve Regularization teknikleri uygulanmıştır.