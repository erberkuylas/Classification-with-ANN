# Yapay Sinir Ağları (YSA) kullanılarak büyük ölçekli bir balık veri setinin sınıflandırılması (Classification of a large-scale fish dataset using ANN)

**Link =>>>> https://www.kaggle.com/code/erberkuylas/classification-with-ann/notebook**

Bu projede 'a large-scale fish dataset' adlı veri setini kullanaraka yapay sinir ağları (ANN) ile balıkların fotoğraflardan sınıflandırmasını sağlayan bir derin öğrenme modeli geliştirdim.Projemi sizlere 3 başlık altında açıklayacağım.Bunlar sırasıyla:

- Kullanılacak kütüphalerinin yüklenmesi ve veri Yükleme ve Etiketleme
- Veri Ön İşleme, Model Oluşturma ve Eğitme İşlemleri
- Sonuçları Değerlendirillmesi ve  Yorumlanması


## Kullanılacak Kütüphalerinin Yüklenmesi ve Veri Setini Çekme İşlemi 
### Projede Kullanılan Kütüphaneler

1. **Pandas**: `import pandas as pd  # Veri işleme ve analiz için`  
2. **NumPy**: `import numpy as np  # Sayısal hesaplamalar ve matris işlemleri için`  
3. **Random**: `import random  # Rastgele sayı üretimi için`  
4. **os**: `import os  # Dosya ve dizin işlemleri için`  
5. **Seaborn**: `import seaborn as sns  # İstatistiksel veri görselleştirme için`  
6. **Matplotlib**: `import matplotlib.pyplot as plt  # Grafik oluşturma ve görselleştirme için`  
7. **train_test_split**: `from sklearn.model_selection import train_test_split  # Veriyi eğitim ve test setlerine ayırmak için`  
8. **LabelEncoder**: `from sklearn.preprocessing import LabelEncoder  # Kategorik verileri sayısal değerlere dönüştürmek için`  
9. **load_img & img_to_array**: `from keras.preprocessing.image import load_img, img_to_array  # Görüntüleri yüklemek ve tensörlere dönüştürmek için`  
10. **confusion_matrix**: `from sklearn.metrics import confusion_matrix  # Karışıklık matrisi oluşturmak için`  
11. **Sequential Model**: `from keras.models import Sequential  # Yapay sinir ağı modeli oluşturmak için`  
12. **Katmanlar (Dense, Flatten, Input, Dropout)**: `from keras.layers import Dense, Flatten, Input, Dropout  # Sinir ağı katmanları tanımlamak için`  
13. **TensorFlow Keras**: `from tensorflow import keras  # Derin öğrenme modelleri için`  
14. **Keras Katmanları**: `from tensorflow.keras import layers  # TensorFlow Keras sinir ağı katmanları için`  
15. **Adam Optimizer**: `from keras.optimizers import Adam  # Model optimizasyon algoritması`  
16. **Pathlib**: `from pathlib import Path  # Dosya yollarını yönetmek için`  
17. **PIL (Python Imaging Library)**: `from PIL import Image  # Görüntü işleme işlemleri için`  
18. **TensorFlow**: `import tensorflow as tf  # Makine öğrenimi ve derin öğrenme için`  
19. **struct**: `import struct  # İkili veri yapılarıyla çalışmak için`  
20. **Warnings**: `import warnings  # Uyarı mesajlarını kontrol etmek için` `warnings.filterwarnings('ignore')  # Uyarıları görmezden gelmek için`


### Veri Yükleme ve Etiketleme
Bu bölümde, **Kaggle'dan bir balık veri setindeki PNG formatındaki resimler yüklenip etiketlenmektedir. `os.walk()` fonksiyonu ile dizinler taranır ve her bir resmin yolu ve etiketi bir listeye eklenir. Ardından bu bilgiler bir **pandas DataFrame**'ine aktarılır.

- `fish_dir`: Balık veri setinin bulunduğu ana dizin.
- `label`: Resmin ait olduğu balık türü (dizin adı).
- `path`: Resmin tam dosya yolu.

Son olarak, `path` ve `label` sütunlarına sahip bir DataFrame oluşturulmuştur.

## Veri Ön İşleme
- Çektiğimi veri setini `head()` , `info()`, `isnull().sum()` gibi fonksiyonlar ile inceliyor ve bilgi ediniyoruz.
- Psta grafiği ile türlerin yüzde kaç dağılıma sahip olduğuna bakıyoruz.

### Eğitim ve Test Verisinin Bölünmesi & Etiket Kodlama 
- Veriyi eğitim ve test setlerine ayırmak için `train_test_split` kullanıldı.
- Eğitim ve test setleri %80 eğitim, %20 test oranında ayrılmıştır.
- `LabelEncoder` kullanılarak etiketler sayısal değerlere dönüştürülmüştür.

### Resimleri Yükleme ve Boyutlandırma
- `yvb` fonksiyonu, verilen dizinden resimleri alır ve belirlenen boyutta yeniden boyutlandırır (varsayılan olarak 224x224).
- Resimler normalleştirilmiştir (`img_to_array` ile 255'e bölünerek).

### Veri Artırma (Data Augmentation)
- `ImageDataGenerator` ile döndürme, kaydırma, zoom ve yatay çevirme gibi veri artırma işlemleri uygulanmıştır.
- Eğitim verisi bu artırma işlemleriyle zenginleştirilmiştir.

### Örnek Resimlerin Görselleştirilmesi
- `plot_images`: Veri setinden rastgele seçilen 16 görüntüyü 4x4 grid düzeninde gösterir.
- Her görüntünün etiketi de görüntüyle birlikte başlık olarak gösterilmektedir.
- Görüntüler matplotlib kullanılarak çizilir.

## Model Oluşturma ve Eğitme 

### Modeli Oluşturma ve Derleme
- **Model Oluşturma**: `Sequential()` modeli oluşturularak katmanlar eklenmiştir.
  - Giriş katmanı: Resimleri düzleştirir (`Flatten`).
  - Gizli katmanlar: 512, 256 ve 128 nörona sahip `Dense` katmanlar kullanılır, her biri ReLU aktivasyonu ile.
  - Çıkış katmanı: Sınıf sayısına bağlı olarak `softmax` aktivasyonu kullanılmıştır.
- **Model Derleme**: Model, Adam optimizasyon algoritması ve `sparse_categorical_crossentropy` kayıp fonksiyonu ile derlenmiştir.

### Early Stopping (Erken Durdurma)
- **Early Stopping**: Doğrulama kaybı (`val_loss`) izlenir ve 5 epoch boyunca iyileşme olmazsa eğitim durdurulur.
- En iyi model ağırlıkları geri yüklenir (`restore_best_weights`).

###  Modeli Eğitme Aşaması
- **Model Eğitimi**: Eğitim ve doğrulama verileriyle model 20 epoch boyunca eğitilir. `EarlyStopping` callback'i ile en iyi sonuçları almak için izleme yapılır.

## Sonuçları Değerlendirillmesi ve  Yorumlanması

### Eğitim Sonuçlarının Görselleştirilmesi
- **Doğruluk Grafiği**: Eğitim ve test doğruluğu zamanla nasıl değiştiğini gösterir.
- **Kayıp Grafiği**: Eğitim ve test kayıplarının epoch'lar boyunca nasıl değiştiğini gösterir.
 ![__results___25_0](https://github.com/user-attachments/assets/2362e531-c7f9-459f-b2b1-bd437315360d)

### Test Seti ile Tahmin Yapma
- **Tahmin Yapma**: Model, test verileri üzerinde tahmin yapar ve tahmin edilen sınıf etiketleri alınır.

### Test Seti Üzerinde Modeli Değerlendirme
- **Model Değerlendirme**: Test kaybı ve doğruluk değerleri hesaplanır.

### F1 Skoru Hesaplama
- **F1 Skoru Hesaplama**: Test verileri üzerindeki tahmin sonuçlarına göre F1 skoru hesaplanır.

### Gerçek ve Tahmin Edilen Etiketler için Karışıklık Matrisi
- **Karışıklık Matrisi**: Gerçek ve tahmin edilen etiketler arasındaki ilişkileri görselleştirir.
- Etiket isimleri ile birlikte görselleştirilmiştir.
  ![__results___29_1](https://github.com/user-attachments/assets/faaa73a9-a7a3-4305-a44c-8c89279dd036)

## Yorumlama 
### Model Performans Değerlendirmesi

Modelin test verisi üzerindeki performansını değerlendirmek için birkaç metrik kullanıldı:

- **Test Loss:** 0.2868  
  Modelin test verisi üzerindeki kaybı düşük seviyede. Bu, modelin tahminlerinin doğru olduğunu ve genel olarak iyi bir performans sergilediğini gösteriyor.

- **Test Accuracy:** 90%  
  Test doğruluğu %90 olarak ölçüldü. Bu, modelin büyük çoğunlukla doğru sınıflandırma yaptığını ve genel olarak etkili bir performans sunduğunu ifade eder.

- **F1 Score:** 0.90  
  F1 skoru, modelin hem hassasiyet hem de duyarlılık açısından dengeli bir performans gösterdiğini belirtir. 0.90'lık bir F1 skoru, modelin yanlış pozitifleri ve yanlış negatifleri iyi yönettiğini gösteriyor.

Sonuç olarak, model oldukça etkili bir performans sergilemekte.










