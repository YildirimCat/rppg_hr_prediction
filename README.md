# rppg_hr_prediction
R-PPG tabanlı, Plane-to-Orthogonal Skin algoritması kullanılarak temassız gerçek zamanlı nabız tahmin etme uygulaması
### Referanslar
+ IEEE paper - "Algorithmic Principles of Remote PPG," W. Wang, A. C. den Brinker, S. Stuijk and G. de Haan. 
+ https://github.com/pavisj/rppg-pos
+ https://www.youtube.com/watch?v=GMN1A8Wfwto
### Başlangıç
Aşağıdaki kodu terminale yazarak gerekli birtakım paketleri ve kütüphaneleri yükleyin. 
```
pip install -r /path/requirements.txt
```
#### Kullanılan Bazı Paketler ve Kütüphanelerin Versiyonları
+ cmake 3.25.0
+ dlib 19.24.0
+ opencv-python 4.5.5.62
+ scikit-learn 0.23.2

### Uygulamanın Çalıştırılması
Gerekli kütüphanelerin ve paketlerin kurulmasından sonra **pos_face_seg.py** dosyasını çalıştırarak uygulamayı kullanabilirsiniz. 

### To-Dos
- [x] Karar matrisinin geliştirilmesi.
- [ ] Blok diyagramın tasarlanması.
- [ ] Akış diyagramının oluşturulması.
- [x] Aktivite diyagramının oluşturulması.
- [ ] Metinsel analizin yazılması.

### Projenin Gelişim Planı
- [x] Literatürdeki kamera ve görüntü işleme algoritmalarını kullanarak gerçek zamanlı temassız nabız ölçümü yapan algoritmaların analiz edilmesi.
- [x] En iyi sonuç veren görüntü işleme algoritmasının tespit edilmesi.
- [ ] Yüz tanıma yazılımın geliştirilmesi.
- [ ] Ara raporun teslim edilmesi.
- [ ] Farklı kişilerden farklı koşullar altında yüz görüntülerinin alınması.
- [ ] Alınan yüz görüntülerinden nabız hesaplamalarının yapılması.
- [ ] Yüz görüntüleri kullanılarak yapılan nabız hesaplamalarının farklı nabız ölçme teknikleriyle elde edilen verilerle mukayese edilmesi.
- [ ] Final raporunun teslim edilmesi ve sunum yapılması.


### Projenin Somut Başarı Kriterleri
- [ ] Yüz görüntüleri kullanılarak yapılan nabız hesaplamalarının nabız ölçme teknikleriyle verilerle en az %95 oranında uyuşması.
- [ ] Belirlenen ilgi bölgesinin (ROI) farklı koşullar altında aynı ve doğru sonuç vermesinin sağlanması.
