# Proje: LOGGPT: Web Trafik Loglarına Dayalı Soru-Cevap (Q&A) Sistemi

## 1. Giriş

Bu projede, bir web sitesinin trafik loglarını kullanarak bir soru-cevap (Q&A) sistemi geliştirilmiştir. Sistem, kullanıcıların doğal dilde sordukları soruları alarak, ilgili log verilerini analiz eder ve bu verilere dayalı en uygun yanıtları oluşturur. Projenin temel amacı, kullanıcı sorularına en alakalı log kayıtlarını bulup, bu kayıtlar üzerinden anlamlı yanıtlar üretebilmektir.

### Kullanılan Teknolojiler ve Modeller

- **Python:** Projenin temel programlama dili olarak kullanılmıştır.
- **Pandas:** Log dosyalarının işlenmesi ve veri manipülasyonu için kullanılmıştır.
- **Faiss:** Log verilerinin vektör bazlı araması için yüksek performanslı bir kütüphane olarak kullanılmıştır.
- **PyTorch & Transformers:** BERT modelini kullanarak log verilerini vektörleştirme ve dil modeli işlemleri için kullanılmıştır.
- **BERT:** Log verilerini vektörleştirmek ve sorguları anlamak için kullanılan önceden eğitilmiş dil modeli.
- **FLAN-T5:** Log verilerine dayalı anlamlı yanıtlar oluşturmak için kullanılan jeneratif dil modeli.
- **Gradio:** Kullanıcı arayüzünü oluşturarak sistemin interaktif bir şekilde kullanılmasını sağlamıştır.

### Projenin Genel Yapısı

Proje, dört ana aşamadan oluşmaktadır:

1. **Veri Hazırlığı ve Ön İşleme:** Web trafik loglarının analizi, temizlenmesi ve vektörlere dönüştürülmesi.
2. **RAG Modelinin Kurulumu:** Retrieval-Augmented Generation (RAG) modelinin kurulumu ve yapılandırılması.
3. **Sistem Entegrasyonu ve Test:** Bilgi alma ve jeneratif modelin entegre edilmesi ve sistemin test edilmesi.
4. **Performans Değerlendirmesi:** Sistemin doğruluğunun ve performansının değerlendirilmesi ve iyileştirme önerileri.

### Sistem Çalışma Durumu

Sistem genel olarak çalışmaktadır, ancak vektörel kütüphaneden kaynaklandığını düşündüğümüz bir sorun nedeniyle, bilgileri getirirken tam olarak doğru sonuçlar üretememektedir. Bu durum, vektör arama işlemlerinde beklenen doğruluğun sağlanmasında bazı zorluklar yaratmaktadır.

## 2. Proje Adımları

### Aşama 1: Veri Hazırlığı ve Ön İşleme

- **Log Kayıtlarının Araştırılması:** İlk olarak, log kayıtları üzerinde araştırmalar yaptım ve Kaggle üzerinden 3GB boyutunda bir log dosyası buldum. Ancak, dosyanın çok büyük olması nedeniyle rastgele seçilen 18.000 veriyi alıp, ayrı bir log dosyasına aktardım.
- **Log Dosyasının Uygun Formata Dönüştürülmesi:** Bu veriyi uygun bir formata çevirmek amacıyla, `log_processor.py` dosyasında regex ifadelerini oluşturdum. Bu süreçte, regexlerin doğru bir şekilde oluşturulmasında yapay zekadan da yardım alarak veriyi düzgün bir CSV formatına dönüştürdüm.

### Aşama 2: Vektörel Database Kurulumu ve Kullanımı

- **Vektörel Database Kullanımı Araştırmaları:** Vektörel database'lerin nasıl çalıştığını bilsem de, burada yeterli deneyimim olmadığı için bu veritabanlarını nasıl kullanabileceğim konusunda araştırmalar yaptım. Bu süreçte en çok faydalandığım kaynaklardan biri freeCodeCamp.org'un YouTube üzerindeki eğitim videoları oldu.
- **Vektörel Database Oluşturma:** Hazır bir kütüphane veya uygulama kullanmak yerine, kendi vektörel database'imi oluşturma kararı aldım ve bunun üzerine araştırmalar yaptım. Bu süreç, en çok zorlandığım kısım oldu, çünkü doğal dilde yapılan sorguları doğru ve alakalı bilgilerle eşleştirmek oldukça zordu.
- **CUDA Desteği ve PyTorch Ayarları:** Büyük dosyalarda daha hızlı işlem yapabilmesi için, PyTorch'u eğer varsa CUDA desteği ile çalışacak şekilde ayarladım. Bu, performansı artırmak için önemli bir adımdı.

### Aşama 3: Sistem Entegrasyonu ve Test

- **Vektörel Database'in Kullanımı ve Sorgu İşlemleri:** Oluşturduğum vektörel database'i kaydettim ve ardından `vectorQuery.py` dosyası üzerinde sorgu işlemlerine başladım. Burada, doğal dili vektörel database'den bilgi getirecek şekilde dönüştürdüm ve sonrasında bu bilgiyi LLM modellerine yorumlatarak kullanıcıya çıktı olarak sundum.
- **LLM Modelleri ile Deneyler:** Başlangıçta GPT-2 modelini denedim ancak bu model fazla ve gereksiz bilgiler ekliyordu. Ayarlarını ne kadar denesem de verimli sonuçlar elde edemedim. Daha sonra Bart modelini kullanmayı denedim ancak yine istediğim verimi alamadım. Son olarak, FLAN-T5 modelini denediğimde, sade ve yeterli bilgi ürettiğini gördüm ve bu modelde sabit kalmayı tercih ettim.

### Aşama 4: Kullanıcı Arayüzü ve Sistem Kullanımı

- **Gradio Arayüzü:** Kullanım kolaylığı sağlamak amacıyla, HuggingFace'in Gradio web arayüzünü kullanarak basit bir arayüz oluşturdum ve sorguları bu arayüzde yaptırdım. 
- **Uygulama Kullanımı:** Uygulama, `logs` klasöründe bulunan `.log` uzantılı dosyayı alıp, `log_processor.py` python dosyası ile gerekli formata çevirir. Daha sonra `vectorQuery.py` python dosyasını çalıştırarak, konsolda verilen localhost üzerinden arayüze erişim sağlanabilir.

### Sistem Performansı

Sistem genel olarak çalışmaktadır, ancak bazı durumlarda vektörel kütüphaneden kaynaklandığını düşündüğümüz performans sorunları ile karşılaşılmaktadır. Bu sorunlar, bilgileri getirirken tam olarak doğru sonuçlar elde edememeye neden olmuştur. Bu, özellikle doğal dilde yapılan sorgularda ve büyük veri setlerinde kendini göstermektedir.

### Örnek Sorgular:

- **Soru:** What is the most used IP address?

- **Faiss Dosyasından Alınan Çıktı:**
    - 188.226.241.38 2019-01-23 14:04:10+03:30 GET /site/enamad HTTP/1.1
    - 178.252.147.90 2019-01-23 14:04:29+03:30 GET /site/enamad HTTP/1.1
    - 37.148.18.46 2019-01-26 19:48:48+03:30 GET /site/enamad HTTP/1.1
    - 46.248.40.93 2019-01-23 14:05:25+03:30 GET /favicon.ico HTTP/1.1
    - 178.252.147.90 2019-01-23 14:05:10+03:30 GET /site/enamad HTTP/1.1
    - 151.239.27.212 2019-01-26 19:49:39+03:30 GET /site/enamad HTTP/1.1
    - 178.252.147.90 2019-01-23 14:05:20+03:30 GET /site/enamad HTTP/1.1
    - 151.239.27.212 2019-01-26 19:49:34+03:30 GET /site/enamad HTTP/1.1
    - 10.252.254.63 2019-01-23 14:04:51+03:30 GET /site/enamad HTTP/1.1
    - 91.185.157.227 2019-01-23 14:04:38+03:30 GET /site/enamad HTTP/1.1

- **LLM Tarafından Yorumlatılan Çıktı:**
    - The most used IP address is 178.252.147.90.

- **Soru:** What is the day with the most traffic?

- **Faiss Dosyasından Alınan Çıktı:**
    - 178.252.147.90 2019-01-23 14:05:20+03:30 GET /site/enamad HTTP/1.1
    - 178.252.147.90 2019-01-23 14:04:29+03:30 GET /site/enamad HTTP/1.1
    - 46.248.40.93 2019-01-23 14:05:25+03:30 GET /favicon.ico HTTP/1.1
    - 178.252.147.90 2019-01-23 14:05:10+03:30 GET /site/enamad HTTP/1.1
    - 188.226.241.38 2019-01-23 14:04:10+03:30 GET /site/enamad HTTP/1.1
    - 151.241.29.102 2019-01-26 19:49:50+03:30 GET /favicon.ico?page=1 HTTP/1.1
    - 46.248.62.102 2019-01-26 19:49:36+03:30 GET /favicon.ico HTTP/1.1
    - 151.239.27.212 2019-01-26 19:49:39+03:30 GET /site/enamad HTTP/1.1
    - 46.224.213.185 2019-01-26 19:49:03+03:30 GET /favicon.ico HTTP/1.1
    - 31.56.207.97 2019-01-23 14:02:49+03:30 GET /favicon.ico HTTP/1.1

- **LLM Tarafından Yorumlatılan Çıktı:**
    - The day with the most traffic is 2019-01-23.


