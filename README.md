# BM201 Proje Ödevi (Görsel işleme özellikli hesap makinesi)

## Yüklemek için:
### Releases
Releases kısmından kendi işletim sisteminiz için olan versiyonu indirebilirsiniz.

### Kaynak kodu kullanarak  
Projeyi bilgisayara kopyaladıktan sonra, (build_no_training branch)

* Terminalde proje klasörüne girin

* "python -m venv ." yada "python3 -m venv ."

* Windows için: ".\HesapMakinesi\Scripts\activate.bat"

* Mac/Linux için: "source HesapMakinesi/bin/activate"

* Pytorch yüklemesi
> Pytorch sitesinden kendi sisteminize göre yükleme komutunu alın (https://pytorch.org/get-started/locally/)  
> PyTorch build : stable  
> Your os: işletim sistemine göre  
> Package: pip  
> Language: python  
> compute platform: cpu   (eğitim için gpu önerilir (nvidia: cuda, AMD: rocm))  
> Oluşan komutu terminalde çalıştırın  

* "pip install -r requirements.txt"

* gui_test.py dosyasını çalıştırarak programı kullanabilirsiniz


## Eğitim hakkında:
https://arxiv.org/abs/2404.10690v2 linkte verilen dataset kullanılmıştır.  

### Kendi eğitiminizi yürütmek için:  
* Main branchi bilgisayara kopyalayın ve yukarıda belirtildiği gibi kurulumu gerçekleştirin  

* Linkte verilen dataseti indirin ve training data klasörüne kopyalayın (./training_data/mathwriting-2024/..) şeklinde

* Ardından preprocess_mathwriting.py scriptini çalıştırabilirsiniz
> scriptin içindeki değişkenleri değitirerek farklı türde veriler oluşturabilirsiniz.  
> SYNTHETIC : Sentetik veri kullanımı +400k daha matematiksel ifade ekler  
> GRID : Arkaya kareli kareli kağıt görünümü verir  
> THICK : Kalın kalem kalınlığı artar


* Ardından eğitim başlatılabilir (train.py)
> train.py dosyasındaki parametreleri düzenlemeyi unutmayın!
