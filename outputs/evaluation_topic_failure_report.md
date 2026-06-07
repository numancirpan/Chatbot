# Evaluation Topic Failure Report

Total failed topics: 16

## akademik_takvim_duyurular

- Fail count: 6
- Classes: no_answer=2, wrong_source=4
- Recommendation: Retrieval yanlis resmi belgeyi one cikariyor. Bu topic icin _candidate_score icinde kaynak basligi, URL oruntusu ve ilgili anahtar terimlere pozitif agirlik; sik karisan topic'lere negatif agirlik eklenmeli.

### akademik_takvim_duyurular_001 (wrong_source)

- Query: Akademik takvim nerede yayınlanır?
- Expected acceptable_source_terms: https://ogrenciisleri.duzce.edu.tr/duyurular, Akademik Takvim, Akademik Takvim / Duyurular
- Returned sources: 2025 2026 Egitim Ogretim Yili Guz Yariyili Final Programi (https://biyomedikal.mf.duzce.edu.tr/duyuru/3776/2025-2026-egitim-ogretim-yili-guz-yariyili-final-programi); Lisans Eğitim-Öğretim ve Sınav Yönetmeliği (https://cdn.duzce.edu.tr/File/GetFile/e61c395b-bd05-4300-b312-0753ab748943#sinav-not-sistemi); Bu belge,güvenli elektronik imza ile imzalanmıştır. (https://cdn.duzce.edu.tr/File/GetFile/06193c82-1968-4c61-9891-d55538ef9528)
- Answer preview: Sayin ogrencimiz, Final sinavi tarihleri akademik takvim ve ilgili birim/OBS duyurulari uzerinden ilan edilir. Kesin tarih ders ve egitim-ogretim yilina gore degistigi icin ilgili yilin akademik takvimi veya OBS sinav programi kontrol edilmelidir.  Dayanak: [K

### akademik_takvim_duyurular_002 (wrong_source)

- Query: Güncel duyuruları nereden takip edebilirim?
- Expected acceptable_source_terms: https://ogrenciisleri.duzce.edu.tr/duyurular, Öğrenci İşleri Duyuruları, Akademik Takvim / Duyurular
- Returned sources: DUZCE UNIVERSITESI KURUM ICI VE KURUMLAR ARASI YATAY GECIS ESASLARI (https://cdn.duzce.edu.tr/File/GetFile/yatay-gecis-yonergesi#basvuru-takvim); Akademik Takvim (https://panel.duzce.edu.tr//Dosya/GetFile/0070ab4a-559e-41a3-b1c7-1c28b68f0514)
- Answer preview: Sayın öğrencimiz, Eski duyuruları takip etmek istiyorsanız, www.duzce.edu.tr adresinden resmi web sitesini ziyaret edebilirsiniz. Bu platformda geçmiş akademik ve idari duyurular yer almaktadır. Ek olarak, Öğrenci İşleri Daire Başkanlığı'nın arşivinde de eski 

### akademik_takvim_duyurular_003 (no_answer)

- Query: Dersler ne zaman başlıyor?
- Expected acceptable_source_terms: https://ogrenciisleri.duzce.edu.tr/duyurular, Akademik Takvim, Akademik Takvim / Duyurular
- Returned sources: Yok
- Answer preview: Sayın öğrencimiz, Bu konuda resmi belgelerde bilgiye ulasilamadi. Lutfen Ogrenci Isleri birimi ile iletisime geciniz.
- Chunk check: FOUND score=10 title=17 Haziran 2021 PERŞEMBE url=https://cdn.duzce.edu.tr/File/GetFile/7aa0f18d-5d69-46f8-a518-3ea6a49203d9
- Chunk match reasons: expected_term:Akademik Takvim, query_terms:dersler,zaman
- Chunk preview: MADDE 12 – (1) İşletmede mesleki eğitimin eğitim ve öğretim dönemlerinde yapılması esastır. İşletmede mesleki eğitim yükseköğretim kurumunun akademik takviminde belirlenen eğitim ve öğretim dönemlerinde başlar ve bu süre

## burs

- Fail count: 6
- Classes: no_answer=3, wrong_source=3
- Recommendation: No-answer vakalarinda chunks.json kanitini kontrol et. Kanit varsa esik, topic metadata veya skorlamayi guclendir; kanit yoksa crawler/kaynak listesi eksik.

### burs_001 (no_answer)

- Query: Burs başvurusu nasıl yapılır?
- Expected acceptable_source_terms: https://panel.duzce.edu.tr//Dosya/GetFile/22a1f570-c7f7-4dd8-8598-616f2ae00e21, Burs Başvuru Duyuruları, Burs
- Returned sources: Yok
- Answer preview: Sayın öğrencimiz, Bu konuda resmi belgelerde bilgiye ulasilamadi. Lutfen Ogrenci Isleri birimi ile iletisime geciniz.
- Chunk check: FOUND score=21 title=Resmi Gazete Tarihi: 28.12.2006 Resmi Gazete Sayısı: 26390 url=https://panel.duzce.edu.tr//Dosya/GetFile/22a1f570-c7f7-4dd8-8598-616f2ae00e21
- Chunk match reasons: expected_url, expected_term:Burs, query_terms:burs
- Chunk preview: MADDE 10 – (1) Devlet üniversitelerinde, bu programlarda öğrenim gören öğrencilerden alınan yurtiçi öğrenim ücretleri, en geç ilgili ayın sonuna kadar özel bütçeye gelir kaydedilmek üzere aktarılır. Aktarılan bu miktar ö

### burs_002 (no_answer)

- Query: Burs almak için hangi şartlar gerekiyor?
- Expected acceptable_source_terms: https://cdn.duzce.edu.tr/File/GetFile/6ed94076-f5da-44ae-aa4a-244b58ad825e, Burs Başvuru Duyuruları, Burs
- Returned sources: Yok
- Answer preview: Sayın öğrencimiz, Bu konuda resmi belgelerde bilgiye ulasilamadi. Lutfen Ogrenci Isleri birimi ile iletisime geciniz.
- Chunk check: FOUND score=21 title=YABANCI UYRUKLU ÖĞRENCİ ADAYLARININ LİSANSÜSTÜ url=https://cdn.duzce.edu.tr/File/GetFile/6ed94076-f5da-44ae-aa4a-244b58ad825e
- Chunk match reasons: expected_url, expected_term:Burs, query_terms:burs
- Chunk preview: MADDE 9- (1) Lisansüstü programlara başvuran adayların başvuru dosyaları, ilgili programın Anabilim/Anasanat Dalı kurulu tarafından değerlendirilir ve yeterli görülen adayların öğrenciliğe kabulleri, ilgili Anabilim/Anas

### burs_003 (wrong_source)

- Query: Burs hangi durumlarda kesilir?
- Expected acceptable_source_terms: https://panel.duzce.edu.tr//Dosya/GetFile/22a1f570-c7f7-4dd8-8598-616f2ae00e21, Burs Başvuru Duyuruları, Burs
- Returned sources: DÜZCE ÜNİVERSİTESİ MÜHENDİSLİK FAKÜLTESİ İŞYERİ EĞİTİMİ (https://panel.duzce.edu.tr//Dosya/GetFile/d1618f0a-1e79-4d1b-9995-3c9dc779a864); TEKNOLOJİ FAKÜLTESİ (https://panel.duzce.edu.tr//Dosya/GetFile/9ee3f7a1-9046-4384-8562-209a701cd98c); Lisans Eğitim-Öğretim ve Sınav Yönetmeliği (https://cdn.duzce.edu.tr/File/GetFile/e61c395b-bd05-4300-b312-0753ab748943#devam-zorunlulugu)
- Answer preview: Sayın öğrencimiz, İşyeri Eğitimine devam etmek zorunludur ve izinli/raporlu olduğu günler haricinde devamsızlık hakları yoktur. Bu durum, Düzce Üniversitesi Mühendislik Fakültesi İşyeri Eğitimi ile ilgili resmi belgede [Kaynak 1] açıkça belirtilmiştir. Devamsı

## muafiyet_intibak

- Fail count: 6
- Classes: no_answer=3, wrong_source=3
- Recommendation: No-answer vakalarinda chunks.json kanitini kontrol et. Kanit varsa esik, topic metadata veya skorlamayi guclendir; kanit yoksa crawler/kaynak listesi eksik.

### muafiyet_intibak_001 (wrong_source)

- Query: Muafiyet başvurusu nasıl yapılır?
- Expected acceptable_source_terms: https://cdn.duzce.edu.tr/File/GetFile/b67441bf-8724-4f5a-b340-a963f8354d6a, Ders Muafiyeti ve İntibak Esasları, Muafiyet / İntibak
- Returned sources: Azami Öğrenim Süresi Sonu Sınav Duyurusu (https://cdn.duzce.edu.tr/File/GetFile/3fde6de1-6503-4417-b064-65c06de89f1d); ORMAN FAKÜLTESİ (https://cdn.duzce.edu.tr/File/GetFile/a771c727-6a6f-4522-aa5a-b441617eaf33); Kurumlararası ve Kurumiçi Yatay Geçiş Başvuruları (Önlisans-Lisans) (https://cdn.duzce.edu.tr/File/GetFile/162cf224-c95a-4393-828f-5f2930500da7)
- Answer preview: Sayın öğrencimiz, Başvurunuz için belirtilen süre, Düzce Üniversitesi Öğrenci İşleri Daire Başkanlığı tarafından belirlenen başvuru kabul edileceği günden en az 15 gün önce olmalıdır. Bu sürenin tam olarak kaç gün olduğu resmi duyuran bir bilgi bulunmamaktadır

### muafiyet_intibak_002 (no_answer)

- Query: İntibak işlemleri neye göre yapılır?
- Expected acceptable_source_terms: https://cdn.duzce.edu.tr/File/GetFile/b67441bf-8724-4f5a-b340-a963f8354d6a, Ders Muafiyeti ve İntibak Esasları, Muafiyet / İntibak
- Returned sources: Yok
- Answer preview: Sayın öğrencimiz, Bu konuda resmi belgelerde bilgiye ulasilamadi. Lutfen Ogrenci Isleri birimi ile iletisime geciniz.
- Chunk check: FOUND score=14 title=Lisans Eğitim-Öğretim ve Sınav Yönetmeliği url=https://cdn.duzce.edu.tr/File/GetFile/b67441bf-8724-4f5a-b340-a963f8354d6a
- Chunk match reasons: expected_url, query_terms:intibak,islemleri
- Chunk preview: MADDE 1- (1) Bu esaslarının amacı; Düzce Üniversitesine kayıt yaptıran öğrencilerinin, daha önce Yükseköğretim Kurulu Başkanlığı tarafından tanınan veya denkliği kabul edilen bir yükseköğretim kurumundan alarak başarılı 

### muafiyet_intibak_003 (wrong_source)

- Query: Muafiyet / intibak için hangi belgeler gerekir?
- Expected acceptable_source_terms: https://cdn.duzce.edu.tr/File/GetFile/b67441bf-8724-4f5a-b340-a963f8354d6a, Ders Muafiyeti ve İntibak Esasları, Muafiyet / İntibak
- Returned sources: Öğrenci Belgesi ve Transkript İşlemleri (https://cdn.duzce.edu.tr/File/GetFile/6ed94076-f5da-44ae-aa4a-244b58ad825e)
- Answer preview: Sayin ogrencimiz, Transkript/not durum belgesi oncelikle e-Devlet Kapisi uzerinden temin edilmelidir. E-Devlet uzerinden temin edilemeyen veya islak imzali/onayli gereken durumlarda ilgili birimin Ogrenci Isleri Burosuna basvurulabilir.  Dayanak: [Kaynak 1] Öğ

## ogrenci_belgesi_transkript

- Fail count: 6
- Classes: no_answer=3, wrong_source=3
- Recommendation: No-answer vakalarinda chunks.json kanitini kontrol et. Kanit varsa esik, topic metadata veya skorlamayi guclendir; kanit yoksa crawler/kaynak listesi eksik.

### ogrenci_belgesi_transkript_001 (wrong_source)

- Query: Öğrenci belgesi nasıl alınır?
- Expected acceptable_source_terms: https://cdn.duzce.edu.tr/File/GetFile/b67441bf-8724-4f5a-b340-a963f8354d6a, Ders Muafiyeti ve İntibak Esasları, Öğrenci Belgesi / Transkript
- Returned sources: Ayrıca, pedagojik formasyon derslerinin alındığı dönemde MEB'e bağlı okullarda ücretli öğretmenlik, özel (https://panel.duzce.edu.tr//Dosya/GetFile/3d303694-adc7-4872-b09f-aa9b2fbb8458); Tek/Çift Ders Sınavları Hakkında SSS (https://tarih.fef.duzce.edu.tr/sayfa/0594/tekcift-ders-sinavlari-hakkinda-sss)
- Answer preview: Sayın öğrencimiz, Ücretli olmayan bir durum söz konusudur. Pedagojik formasyon derslerinin alındığı dönemde MEB'e bağlı okullarda ücretli öğretmenlik yapma imkanı bulunmamaktadır. Eğer tek/cift ders sınavlarına katılmanız gerekiyorsa, ilgili bölüm başkanına ba

### ogrenci_belgesi_transkript_002 (wrong_source)

- Query: Transkript nereden alınır?
- Expected acceptable_source_terms: https://cdn.duzce.edu.tr/File/GetFile/5c5910a2-7cf3-475c-86ff-83539c7b73da, Kayıt Yenileme İşlemleri, Öğrenci Belgesi / Transkript
- Returned sources: Öğrenci Belgesi ve Transkript İşlemleri (https://mf.duzce.edu.tr/sayfa/8955/cift-anadal-programlari)
- Answer preview: Sayin ogrencimiz, Ogrenci belgesi ve transkript/not durum belgesi oncelikle e-Devlet Kapisi uzerinden temin edilmelidir. Kaynakta e-Devlet uzerinden alinan belgelerin kamu kurumlari tarafindan kabul edilmesi gerektigi belirtilmektedir. E-Devlet uzerinden temin

### ogrenci_belgesi_transkript_003 (no_answer)

- Query: İngilizce transkript alabilir miyim?
- Expected acceptable_source_terms: https://cdn.duzce.edu.tr/File/GetFile/b67441bf-8724-4f5a-b340-a963f8354d6a, Ders Muafiyeti ve İntibak Esasları, Öğrenci Belgesi / Transkript
- Returned sources: Yok
- Answer preview: Sayın öğrencimiz, Bu konuda resmi belgelerde bilgiye ulasilamadi. Lutfen Ogrenci Isleri birimi ile iletisime geciniz.
- Chunk check: FOUND score=13 title=Lisans Eğitim-Öğretim ve Sınav Yönetmeliği url=https://cdn.duzce.edu.tr/File/GetFile/b67441bf-8724-4f5a-b340-a963f8354d6a
- Chunk match reasons: expected_url, query_terms:transkript
- Chunk preview: MADDE 3- (1) Bu esaslarda geçen, a) AKTS: Avrupa Kredi Transfer Sistemini (ECTS), b) Birim: Düzce Üniversitesinin fakülte/yüksekokul/meslek yüksekokulunu, c) Bölüm: Düzce Üniversitesinin fakülte/yüksekokul/meslek yükseko

## disiplin_islemleri

- Fail count: 5
- Classes: no_answer=1, wrong_source=4
- Recommendation: Retrieval yanlis resmi belgeyi one cikariyor. Bu topic icin _candidate_score icinde kaynak basligi, URL oruntusu ve ilgili anahtar terimlere pozitif agirlik; sik karisan topic'lere negatif agirlik eklenmeli.

### disiplin_islemleri_002 (wrong_source)

- Query: Disiplin süreci nasıl işler?
- Expected acceptable_source_terms: https://panel.duzce.edu.tr//Dosya/GetFile/d1618f0a-1e79-4d1b-9995-3c9dc779a864, Öğrenci Disiplin Yönetmeliği, Disiplin İşlemleri
- Returned sources: Yükseköğretim Kurumlarında kayıtlı öğrencilerinin disiplin işleri aşağıda verilen (https://cdn.duzce.edu.tr/File/GetFile/a3dc4e5a-a9d6-45f0-8ec8-0b6d8293987c); 2547 SAYILI YÜKSEKÖĞRETİM KANUNUNUN 54 ÜNCÜ MADDESİNE GÖRE (https://cdn.duzce.edu.tr/File/GetFile/393d1bfd-1e64-4f0f-8c91-9f040cbbf0fa)
- Answer preview: Sayın öğrencimiz, Savunma hakkınız bulunmaktadır. Yükseköğretim Kanunu 54.maddesine göre, disiplin soruşturması başlatılan öğrenciye isnat edilen suçun neden ibaret olduğu ve savunmasını yapacağı tarihten en az yedi gün önce yazılı olarak bildirilmesi gerekmek

### disiplin_islemleri_003 (wrong_source)

- Query: Disiplin kararına itiraz edilebilir mi?
- Expected acceptable_source_terms: https://cdn.duzce.edu.tr/File/GetFile/b67441bf-8724-4f5a-b340-a963f8354d6a, Ders Muafiyeti ve İntibak Esasları, Disiplin İşlemleri
- Returned sources: 2547 SAYILI YÜKSEKÖĞRETİM KANUNUNUN 54 ÜNCÜ MADDESİNE GÖRE (https://cdn.duzce.edu.tr/File/GetFile/672fbe98-7a69-4fa4-905e-dd79db4244a3); Yükseköğretim Kurumlarında kayıtlı öğrencilerinin disiplin işleri aşağıda verilen (https://cdn.duzce.edu.tr/File/GetFile/a3dc4e5a-a9d6-45f0-8ec8-0b6d8293987c)
- Answer preview: Sayın öğrencimiz, Soruşturmacı tarafından soruşturma istemi yazısı tebliğ edildiği güne kadar en az 7 gün süre verilmelidir. Bu süre sonunda savunma yapılmamışsa, durum değerlendirilerek yeni bir tarih belirlenmesi beklenmektedir. Bu konuda resmi belgelerde bu

### disiplin_islemleri_004 (wrong_source)

- Query: Kınama cezası hangi fiiller için verilir?
- Expected acceptable_source_terms: https://cdn.duzce.edu.tr/File/GetFile/a3dc4e5a-a9d6-45f0-8ec8-0b6d8293987c, Öğrenci Disiplin Yönetmeliği, Disiplin İşlemleri
- Returned sources: Azami Öğrenim Süresi Sonu Sınav Duyurusu (https://cdn.duzce.edu.tr/File/GetFile/3fde6de1-6503-4417-b064-65c06de89f1d); ÖZEL GEREKSİNİMLİ ÖĞRENCİLER İÇİN EĞİTİMDE FIRSAT EŞİTLİĞİ (https://panel.duzce.edu.tr//Dosya/GetFile/024ef66a-d5d2-472f-86df-443989aaf1d1)
- Answer preview: Sayın öğrencimiz, Resmi kaynaklarda yer alan ilgili bilgi asagidadir: - e) DGS: Ölçme, Seçme ve Yerleştirme Merkezi tarafından yapılan dikey geçiş sınavını, f) Ek süre sonu sınavları: Ek süre sonunda başarısız dersi bulunan öğrencilerin gireceği sınavları, g) 

## harc_ucret

- Fail count: 5
- Classes: no_answer=2, wrong_source=3
- Recommendation: Retrieval yanlis resmi belgeyi one cikariyor. Bu topic icin _candidate_score icinde kaynak basligi, URL oruntusu ve ilgili anahtar terimlere pozitif agirlik; sik karisan topic'lere negatif agirlik eklenmeli.

### harc_ucret_001 (no_answer)

- Query: Harç ne zaman yatırılır?
- Expected acceptable_source_terms: https://cdn.duzce.edu.tr/File/GetFile/c4035fe1-0ea7-42b8-a7dc-8929456f9745, Katkı Payı ve Öğrenim Ücreti, Harç / Ücret
- Returned sources: Yok
- Answer preview: Sayın öğrencimiz, Bu konuda resmi belgelerde bilgiye ulasilamadi. Lutfen Ogrenci Isleri birimi ile iletisime geciniz.
- Chunk check: FOUND score=13 title=DÜZCE ÜNİVERSİTESİ İKİNCİ ÖĞRETİMDE İLK YÜZDE ONA GİREN url=https://cdn.duzce.edu.tr/File/GetFile/c4035fe1-0ea7-42b8-a7dc-8929456f9745
- Chunk match reasons: expected_url, query_terms:zaman
- Chunk preview: e) Üst yarıyıldan alınan ve başarısız olunan dersler, bir öğrencinin mevcut yarıyılda sıralama listesine girmesine engel teşkil etmez. Bu dersler, öğrenci ilgili üst yarıyıla geldiği zaman sıralama listesi belirlenirken 

### harc_ucret_002 (no_answer)

- Query: Öğrenim ücreti ne kadar?
- Expected acceptable_source_terms: https://cdn.duzce.edu.tr/File/GetFile/c4035fe1-0ea7-42b8-a7dc-8929456f9745, Katkı Payı ve Öğrenim Ücreti, Harç / Ücret
- Returned sources: Yok
- Answer preview: Sayın öğrencimiz, Bu konuda resmi belgelerde bilgiye ulasilamadi. Lutfen Ogrenci Isleri birimi ile iletisime geciniz.
- Chunk check: FOUND score=14 title=DÜZCE ÜNİVERSİTESİ İKİNCİ ÖĞRETİMDE İLK YÜZDE ONA GİREN url=https://cdn.duzce.edu.tr/File/GetFile/c4035fe1-0ea7-42b8-a7dc-8929456f9745
- Chunk match reasons: expected_url, query_terms:ogrenim,ucreti
- Chunk preview: öğrencilerinden cari hizmet maliyetlerine öğrenci katkısı olarak alınacak miktarını, g) Öğrenci: Düzce Üniversitesinde ön lisans ve lisans düzeyinde eğitim-öğretim gören ikinci öğretim programı öğrencisini, ğ) Öğrenim Üc

### harc_ucret_003 (wrong_source)

- Query: Harç yatırmadan ders kaydı yapılır mı?
- Expected acceptable_source_terms: https://mf.duzce.edu.tr/sayfa/8a10/ders-kayitlari-esnasinda-sikca-sorulan-sorular-ve-cevaplari, Ders Kayıtları Esnasında Sıkça Sorulan Sorular ve Cevapları, Harç / Ücret
- Returned sources: Lisans Eğitim-Öğretim ve Sınav Yönetmeliği (https://cdn.duzce.edu.tr/File/GetFile/e61c395b-bd05-4300-b312-0753ab748943); Ders Kayıt ve Kayıt Yenileme (https://cdn.duzce.edu.tr/File/GetFile/1ed55517-ee0c-431a-a214-7a41d74eaa70)
- Answer preview: Sayin ogrencimiz, Ders kaydinin kesinlesmesi icin danisman onayi sureci takip edilmelidir. 2025-2026 bahar yariyili duyurusunda danisman onaylari 4 Subat 2026 - 15 Subat 2026 arasi olarak ilan edilmistir.  Dayanak: [Kaynak 1] Lisans Eğitim-Öğretim ve Sınav Yön

## not_sistemi_ortalama

- Fail count: 5
- Classes: no_answer=3, wrong_source=2
- Recommendation: No-answer vakalarinda chunks.json kanitini kontrol et. Kanit varsa esik, topic metadata veya skorlamayi guclendir; kanit yoksa crawler/kaynak listesi eksik.

### not_sistemi_ortalama_002 (no_answer)

- Query: DC ile geçilir mi?
- Expected acceptable_source_terms: https://cdn.duzce.edu.tr/File/GetFile/3fde6de1-6503-4417-b064-65c06de89f1d, Lisans Eğitim Öğretim Yönetmeliği, Not Sistemi / Ortalama
- Returned sources: Yok
- Answer preview: Sayın öğrencimiz, Bu konuda resmi belgelerde bilgiye ulasilamadi. Lutfen Ogrenci Isleri birimi ile iletisime geciniz.
- Chunk check: FOUND score=12 title=DÜZCE ÜNİVERSİTESİ AZAMİ ÖĞRENİM SÜRESİ SONUNDA YAPILACAK url=https://cdn.duzce.edu.tr/File/GetFile/3fde6de1-6503-4417-b064-65c06de89f1d
- Chunk match reasons: expected_url
- Chunk preview: DÜZCE ÜNİVERSİTESİ AZAMİ ÖĞRENİM SÜRESİ SONUNDA YAPILACAK İŞLEMLERE İLİŞKİN YÖNERGE BİRİNCİ BÖLÜM Başlangıç Hükümleri Amaç ve kapsam

### not_sistemi_ortalama_003 (wrong_source)

- Query: FF alınca ne olur?
- Expected acceptable_source_terms: https://cdn.duzce.edu.tr/File/GetFile/06193c82-1968-4c61-9891-d55538ef9528, Lisans Eğitim Öğretim Yönetmeliği, Not Sistemi / Ortalama
- Returned sources: Azami Öğrenim Süresi Sonu Sınav Duyurusu (https://cdn.duzce.edu.tr/File/GetFile/3fde6de1-6503-4417-b064-65c06de89f1d)
- Answer preview: Sayın öğrencimiz, Bütten giriş hakkınız için aşağıdaki koşullardan birini karşılamalısınız: 1. Mezuniyetleri için gerekli AGNO’yu sağlayamayan ve azami süresini tamamlayan öğrenciler, öncelikle “DD” veya “DC” harf notu aldıkları derslerden başlamak şartıyla is

### not_sistemi_ortalama_004 (wrong_source)

- Query: AGNO ve GANO aynı şey mi?
- Expected acceptable_source_terms: https://cdn.duzce.edu.tr/File/GetFile/e61c395b-bd05-4300-b312-0753ab748943, Düzce Üniversitesi Lisans Eğitim-Öğretim ve Sınav Yönetmeliği, Not Sistemi / Ortalama
- Returned sources: DÜZCE ÜNİVERSİTESİ ORMAN FAKÜLTESİ (https://cdn.duzce.edu.tr/File/GetFile/4a2f99cf-bc06-4a37-9746-f847178d0117)
- Answer preview: Sayın öğrencimiz, Dönem ortalamaları jüri üyelerinin ayrı ayrı verdikleri notların ortalamasından elde edilir. Bu bilgi [Kaynak 1] sayesinde doğrudan belgeden alınıyor. Eğer daha fazla sorunuz varsa, lütfen bize bildiriniz. Yardımcılığımız için memnuniyetle ba

## add_drop

- Fail count: 4
- Classes: no_answer=1, wrong_source=3
- Recommendation: Retrieval yanlis resmi belgeyi one cikariyor. Bu topic icin _candidate_score icinde kaynak basligi, URL oruntusu ve ilgili anahtar terimlere pozitif agirlik; sik karisan topic'lere negatif agirlik eklenmeli.

### add_drop_001 (wrong_source)

- Query: Add-drop haftası ne zaman?
- Expected acceptable_source_terms: https://cdn.duzce.edu.tr/File/GetFile/264a1c72-d5aa-4885-961e-75e0ba94acd5, 2024-2025 Eğitim-Öğretim Yılı Akademik Takvimi, Add-Drop
- Returned sources: Yatay Geçiş (https://cdn.duzce.edu.tr/File/GetFile/8c0c9acc-09db-4c71-bfac-8aeec6768d6d)
- Answer preview: Sayın öğrencimiz, Sorunuzda belirtilen son günde hangi tarihten bahsettiğini merak ediyorum. Eğer dönem sonu sınavları veya tez savunma gibi genel bir konuda sorularınızı ortaya koymaktaydsanız, 2026-2027 Eğitim-Yönetim Yılı için belgelerdeki tarihler şu şekil

### add_drop_003 (no_answer)

- Query: Seçtiğim dersi bırakabilir miyim?
- Expected acceptable_source_terms: https://cdn.duzce.edu.tr/File/GetFile/e61c395b-bd05-4300-b312-0753ab748943, Düzce Üniversitesi Lisans Eğitim-Öğretim ve Sınav Yönetmeliği, Add-Drop
- Returned sources: Yok
- Answer preview: Sayın öğrencimiz, Bu konuda resmi belgelerde bilgiye ulasilamadi. Lutfen Ogrenci Isleri birimi ile iletisime geciniz.
- Chunk check: FOUND score=21 title=Lisans Eğitim-Öğretim ve Sınav Yönetmeliği url=https://cdn.duzce.edu.tr/File/GetFile/e61c395b-bd05-4300-b312-0753ab748943#devam-zorunlulugu
- Chunk match reasons: expected_url, expected_term:Düzce Üniversitesi Lisans Eğitim-Öğretim ve Sınav Yönetmeliğ, query_terms:dersi
- Chunk preview: DUZCE UNIVERSITESI LISANS EGITIM OGRETIM VE SINAV YONETMELIGI - DEVAM ZORUNLULUGU Ilk defa alinan derslerde ogrencilerin teorik derslerin en az yuzde 70'ine, uygulamali derslerin en az yuzde 80'ine devam etmesi gerekir. 

### add_drop_004 (wrong_source)

- Query: Add-drop işlemleri hangi gün sona erer?
- Expected acceptable_source_terms: https://cdn.duzce.edu.tr/File/GetFile/264a1c72-d5aa-4885-961e-75e0ba94acd5, 2024-2025 Eğitim-Öğretim Yılı Akademik Takvimi, Add-Drop
- Returned sources: Kurumlararası ve Kurumiçi Yatay Geçiş Başvuruları (Önlisans-Lisans) (https://cdn.duzce.edu.tr/File/GetFile/daf5f5c5-7834-41a5-9267-325b69b3ed05)
- Answer preview: Sayın öğrencimiz, Son gün ve saatin belirlenmesine ilişkin resmi bilgilerde açık bir süre veya sınır bulunmamaktadır. Ancak genel olarak sistem üzerinden girilen tüm işlemler için geçerli olan en az 15 dakika süresi vardır. Bu konuda daha fazla bilgiye ihtiyac

## askerlik_tecili

- Fail count: 4
- Classes: no_answer=3, wrong_source=1
- Recommendation: No-answer vakalarinda chunks.json kanitini kontrol et. Kanit varsa esik, topic metadata veya skorlamayi guclendir; kanit yoksa crawler/kaynak listesi eksik.

### askerlik_tecili_001 (no_answer)

- Query: Askerlik tecili nasıl yapılır?
- Expected acceptable_source_terms: https://panel.duzce.edu.tr//Dosya/GetFile/3d303694-adc7-4872-b09f-aa9b2fbb8458, Askerlik Tecil İşlemleri, Askerlik Tecili
- Returned sources: Yok
- Answer preview: Sayın öğrencimiz, Bu konuda resmi belgelerde bilgiye ulasilamadi. Lutfen Ogrenci Isleri birimi ile iletisime geciniz.
- Chunk check: FOUND score=21 title=T.C. url=https://panel.duzce.edu.tr//Dosya/GetFile/3d303694-adc7-4872-b09f-aa9b2fbb8458
- Chunk match reasons: expected_url, expected_term:Askerlik Tecil İşlemleri, query_terms:askerlik
- Chunk preview: öğretim kurumlarında sözleşmeli öğretmenlik veya Diyanet İşleri Başkanlığı'na bağlı olarak Kur'an Kursu Öğreticiliği yapmakta olanlar, Öğretmenlik Uygulaması dersinden muaf tutulabilir. (5) Programda aldığı dersleri, öng

### askerlik_tecili_002 (wrong_source)

- Query: Tecil için hangi belgeler gerekir?
- Expected acceptable_source_terms: https://panel.duzce.edu.tr//Dosya/GetFile/3d303694-adc7-4872-b09f-aa9b2fbb8458, Askerlik Tecil İşlemleri, Askerlik Tecili
- Returned sources: Öğrenci Belgesi ve Transkript İşlemleri (https://cdn.duzce.edu.tr/File/GetFile/6ed94076-f5da-44ae-aa4a-244b58ad825e)
- Answer preview: Sayin ogrencimiz, Ogrenci belgesi oncelikle e-Devlet Kapisi uzerinden temin edilmelidir. E-Devlet uzerinden temin edilemeyen veya islak imzali/onayli gerekli olan belgeler icin ilgili birimin Ogrenci Isleri Burosuna basvurulabilir.  Dayanak: [Kaynak 1] Öğrenci

### askerlik_tecili_003 (no_answer)

- Query: Tecili öğrenci işleri mi yapıyor?
- Expected acceptable_source_terms: https://panel.duzce.edu.tr//Dosya/GetFile/3d303694-adc7-4872-b09f-aa9b2fbb8458, Askerlik Tecil İşlemleri, Askerlik Tecili
- Returned sources: Yok
- Answer preview: Sayın öğrencimiz, Bu konuda resmi belgelerde bilgiye ulasilamadi. Lutfen Ogrenci Isleri birimi ile iletisime geciniz.
- Chunk check: FOUND score=22 title=T.C. url=https://panel.duzce.edu.tr//Dosya/GetFile/3d303694-adc7-4872-b09f-aa9b2fbb8458
- Chunk match reasons: expected_url, expected_term:Askerlik Tecil İşlemleri, query_terms:ogrenci,isleri
- Chunk preview: öğretim kurumlarında sözleşmeli öğretmenlik veya Diyanet İşleri Başkanlığı'na bağlı olarak Kur'an Kursu Öğreticiliği yapmakta olanlar, Öğretmenlik Uygulaması dersinden muaf tutulabilir. (5) Programda aldığı dersleri, öng

## ders_kaydi

- Fail count: 4
- Classes: no_answer=0, wrong_source=4
- Recommendation: Retrieval yanlis resmi belgeyi one cikariyor. Bu topic icin _candidate_score icinde kaynak basligi, URL oruntusu ve ilgili anahtar terimlere pozitif agirlik; sik karisan topic'lere negatif agirlik eklenmeli.

### ders_kaydi_001 (wrong_source)

- Query: Kayıt yenileme nasıl yapılır?
- Expected acceptable_source_terms: https://cdn.duzce.edu.tr/File/GetFile/e61c395b-bd05-4300-b312-0753ab748943, Düzce Üniversitesi Lisans Eğitim-Öğretim ve Sınav Yönetmeliği, Ders Kaydı / Kayıt Yenileme
- Returned sources: YABANCI UYRUKLU ÖĞRENCİ ADAYLARININ LİSANSÜSTÜ (https://cdn.duzce.edu.tr/File/GetFile/6ed94076-f5da-44ae-aa4a-244b58ad825e); DÜZCE ÜNİVERSİTESİ İKİNCİ ÖĞRETİMDE İLK YÜZDE ONA GİREN (https://cdn.duzce.edu.tr/File/GetFile/c4035fe1-0ea7-42b8-a7dc-8929456f9745)
- Answer preview: Sayın öğrencimiz, Yabancı uyruklu öğrencilerin harç ödemesi zorunludur. Ödeme tutarı EYK teklifi ve Üniversite Yönetim Kurulu kararıyla belirlenir [Kaynak 1]. Eğer daha fazla bilgiye ihtiyacınız varsa, lütfen Öğrenci İşleri birimimize ulaşabilirsiniz.  Dayanak

### ders_kaydi_003 (wrong_source)

- Query: Bir dönemde maksimum kaç kredi alınabilir?
- Expected acceptable_source_terms: https://cdn.duzce.edu.tr/File/GetFile/e61c395b-bd05-4300-b312-0753ab748943, Düzce Üniversitesi Lisans Eğitim-Öğretim ve Sınav Yönetmeliği, Ders Kaydı / Kayıt Yenileme
- Returned sources: Bilgisayar Mühendisliği - Staj SSS (https://bm.mf.duzce.edu.tr/sayfa/878b/stajlar-hakkinda-sikca-sorulan-sorular)
- Answer preview: Sayın öğrencimiz, Minimum kredi sınırı konusunda resmi belgelerde açık bir değil olduğuna göre, bu konuda resmi belgelerden bilgiye ulaşamadık. Lütfen Öğrenci İşleri Birimi ile iletişime geçiniz. Bu konu hakkında daha fazla bilgi almak isteyebilirseniz veya öz

### ders_kaydi_004 (wrong_source)

- Query: Ders kaydında danışman onayı gerekiyor mu?
- Expected acceptable_source_terms: https://cdn.duzce.edu.tr/File/GetFile/e61c395b-bd05-4300-b312-0753ab748943, Düzce Üniversitesi Lisans Eğitim-Öğretim ve Sınav Yönetmeliği, Ders Kaydı / Kayıt Yenileme
- Returned sources: 2025-2026 Bahar Ders Kayıt ve Kayıt Yenileme Duyurusu (https://ef.duzce.edu.tr/duyuru/6771/2025-2026-egitim-ogretim-yili-bahar-yariyili-ders-kayit-ve-kayit-yenileme-duyurusu#ekle-sil-akts); Ders Kayıt ve Kayıt Yenileme (https://cdn.duzce.edu.tr/File/GetFile/1ed55517-ee0c-431a-a214-7a41d74eaa70); Akademik Takvim (https://panel.duzce.edu.tr//Dosya/GetFile/f8f894c3-b779-466b-9054-c0bd90c74b05)
- Answer preview: Sayin ogrencimiz, Ders secimi OBS uzerinden yapilir ve danisman onayi sureci takip edilir. Harc/katki payi odemesi gereken ogrenciler ders kaydi oncesinde odemelerini tamamlamalidir.  Dayanak: [Kaynak 1] 2025-2026 Bahar Ders Kayıt ve Kayıt Yenileme Duyurusu; [

## yaz_okulu

- Fail count: 4
- Classes: no_answer=2, wrong_source=2
- Recommendation: No-answer vakalarinda chunks.json kanitini kontrol et. Kanit varsa esik, topic metadata veya skorlamayi guclendir; kanit yoksa crawler/kaynak listesi eksik.

### yaz_okulu_002 (no_answer)

- Query: Yaz okulunda en fazla kaç ders alınabilir?
- Expected acceptable_source_terms: https://cdn.duzce.edu.tr/File/GetFile/0a10d86d-7915-4164-b60b-2e4d3ac59f19, Yaz Okulu Uygulama Esasları, Yaz Okulu
- Returned sources: Yok
- Answer preview: Sayın öğrencimiz, Bu konuda resmi belgelerde bilgiye ulasilamadi. Lutfen Ogrenci Isleri birimi ile iletisime geciniz.
- Chunk check: FOUND score=23 title=Yaz Okulu Yönergesi url=https://cdn.duzce.edu.tr/File/GetFile/0a10d86d-7915-4164-b60b-2e4d3ac59f19
- Chunk match reasons: expected_url, expected_term:Yaz Okulu, query_terms:okulunda,fazla,ders
- Chunk preview: MADDE 9 - (1) (Değişik 11.07.2025 tarihli 2025/87 sayılı Senato Kararı) Yaz okulunda öğretim süresi, yaz okulu sonu sınavları hariç en az beş en fazla yedi haftadır. Yaz okulunda açılan her ders için, normal dönemde yapı

### yaz_okulu_004 (wrong_source)

- Query: Yaz okulunda devam zorunluluğu var mı?
- Expected acceptable_source_terms: https://cdn.duzce.edu.tr/File/GetFile/0a10d86d-7915-4164-b60b-2e4d3ac59f19, Yaz Okulu Uygulama Esasları, Yaz Okulu
- Returned sources: Öğrenci Belgesi ve Transkript İşlemleri (https://cdn.duzce.edu.tr/File/GetFile/5c5910a2-7cf3-475c-86ff-83539c7b73da); Lisans Eğitim-Öğretim ve Sınav Yönetmeliği (https://cdn.duzce.edu.tr/File/GetFile/e61c395b-bd05-4300-b312-0753ab748943#devam-zorunlulugu); Amaç (https://cdn.duzce.edu.tr/File/GetFile/8baa2513-ea98-4920-b5e8-d7ac77145b43)
- Answer preview: Sayın öğrencimiz, Uygulamalı derslerin devamsızlığı zorunluluğu hakkında soru sormaktan memnuniyet duyarım. Öğrenci belgesinde ve transkript işlemlerinden bilgi edinebiliriz ki, uygulama bölümüne %80'ye kadar devam etmek zorundasınız [Kaynak 1]. Bu durum değiş

### yaz_okulu_005 (no_answer)

- Query: Yaz okulunda en fazla kaç AKTS alınabilir?
- Expected acceptable_source_terms: https://cdn.duzce.edu.tr/File/GetFile/0a10d86d-7915-4164-b60b-2e4d3ac59f19, Yaz Okulu Uygulama Esasları, Yaz Okulu
- Returned sources: Yok
- Answer preview: Sayın öğrencimiz, Bu konuda resmi belgelerde bilgiye ulasilamadi. Lutfen Ogrenci Isleri birimi ile iletisime geciniz.
- Chunk check: FOUND score=22 title=Yaz Okulu Yönergesi url=https://cdn.duzce.edu.tr/File/GetFile/0a10d86d-7915-4164-b60b-2e4d3ac59f19
- Chunk match reasons: expected_url, expected_term:Yaz Okulu, query_terms:okulunda,akts
- Chunk preview: MADDE 7 - (1) Yaz okulunda ders alacak öğrenci; yaz okulu kesin kayıt işlemlerini yaz okulu dersleri başlamadan akademik takvimde belirlenecek tarihler içinde yapar. Öğrencilerin yaz okulu ile ilgili kayıt işlemleri, ger

## devamsizlik

- Fail count: 3
- Classes: no_answer=2, wrong_source=1
- Recommendation: No-answer vakalarinda chunks.json kanitini kontrol et. Kanit varsa esik, topic metadata veya skorlamayi guclendir; kanit yoksa crawler/kaynak listesi eksik.

### devamsizlik_003 (no_answer)

- Query: Uygulamalı derslerde devamsızlık nasıl hesaplanır?
- Expected acceptable_source_terms: https://cdn.duzce.edu.tr/File/GetFile/5c5910a2-7cf3-475c-86ff-83539c7b73da, Kayıt Yenileme İşlemleri, Devamsızlık
- Returned sources: Yok
- Answer preview: Sayın öğrencimiz, Bu konuda resmi belgelerde bilgiye ulasilamadi. Lutfen Ogrenci Isleri birimi ile iletisime geciniz.
- Chunk check: FOUND score=30 title=Öğrenci Belgesi ve Transkript İşlemleri url=https://cdn.duzce.edu.tr/File/GetFile/5c5910a2-7cf3-475c-86ff-83539c7b73da
- Chunk match reasons: expected_url, expected_term:Kayıt Yenileme İşlemleri, expected_term:Devamsızlık, query_terms:derslerde,devamsizlik
- Chunk preview: MADDE 7- (1) Öğrenciler, akademik takvimde belirtilen süre içerisinde öğrenim katkı payı yarıyıl taksitini yatırarak ve alacağı dersleri seçerek internet üzerinden kayıt yenileme işlemlerini yaparlar. Ders kaydı danışman

### devamsizlik_004 (no_answer)

- Query: Teorik ve uygulamalı derslerde devamsızlık sınırı aynı mı?
- Expected acceptable_source_terms: https://cdn.duzce.edu.tr/File/GetFile/e61c395b-bd05-4300-b312-0753ab748943, Düzce Üniversitesi Lisans Eğitim-Öğretim ve Sınav Yönetmeliği, Devamsızlık
- Returned sources: Yok
- Answer preview: Sayın öğrencimiz, Bu konuda resmi belgelerde bilgiye ulasilamadi. Lutfen Ogrenci Isleri birimi ile iletisime geciniz.
- Chunk check: FOUND score=32 title=Lisans Eğitim-Öğretim ve Sınav Yönetmeliği url=https://cdn.duzce.edu.tr/File/GetFile/e61c395b-bd05-4300-b312-0753ab748943#devam-zorunlulugu
- Chunk match reasons: expected_url, expected_term:Düzce Üniversitesi Lisans Eğitim-Öğretim ve Sınav Yönetmeliğ, expected_term:Devamsızlık, query_terms:teorik,uygulamali,derslerde,devamsizlik
- Chunk preview: DUZCE UNIVERSITESI LISANS EGITIM OGRETIM VE SINAV YONETMELIGI - DEVAM ZORUNLULUGU Ilk defa alinan derslerde ogrencilerin teorik derslerin en az yuzde 70'ine, uygulamali derslerin en az yuzde 80'ine devam etmesi gerekir. 

### devamsizlik_005 (wrong_source)

- Query: Devamsızlıktan kalan öğrenci finale girebilir mi?
- Expected acceptable_source_terms: https://cdn.duzce.edu.tr/File/GetFile/3fde6de1-6503-4417-b064-65c06de89f1d, Lisans Eğitim Öğretim Yönetmeliği, Devamsızlık
- Returned sources: 2025-2026 Bahar Ders Kayıt ve Kayıt Yenileme Duyurusu (https://ef.duzce.edu.tr/duyuru/6771/2025-2026-egitim-ogretim-yili-bahar-yariyili-ders-kayit-ve-kayit-yenileme-duyurusu#ekle-sil-akts); Ders Kayıt ve Kayıt Yenileme (https://cdn.duzce.edu.tr/File/GetFile/1ed55517-ee0c-431a-a214-7a41d74eaa70)
- Answer preview: Sayın öğrencimiz, OBS'de devamsız görünmek durumunda, lütfen Öğrenci İşleri Daire Başkanlığı'na başvurun ve ilgili belgelerinizi sunarak durumu açıklayınız. Bu süreçte size yardımcı olmak için buradayım.  Dayanak: [Kaynak 1] 2025-2026 Bahar Ders Kayıt ve Kayıt

## mezuniyet

- Fail count: 3
- Classes: no_answer=1, wrong_source=2
- Recommendation: Retrieval yanlis resmi belgeyi one cikariyor. Bu topic icin _candidate_score icinde kaynak basligi, URL oruntusu ve ilgili anahtar terimlere pozitif agirlik; sik karisan topic'lere negatif agirlik eklenmeli.

### mezuniyet_002 (wrong_source)

- Query: Mezun olmak için ortalama kaç olmalı?
- Expected acceptable_source_terms: https://cdn.duzce.edu.tr/File/GetFile/2b51265c-b4f6-4ba4-b61d-7a8f4c6dd0ff, Lisans Eğitim Öğretim Yönetmeliği, Mezuniyet
- Returned sources: Düzce Üniversitesi Mevzuat Komisyonu (https://cdn.duzce.edu.tr/File/GetFile/7772123f-c7c5-4552-b300-7147262a5377); CAP ve Yandal (https://bm.mf.duzce.edu.tr/sayfa/fdcb/cift-anadal-ve-yandal)
- Answer preview: Sayın öğrencimiz, Tüm derslerinizi geçmiş olması zorunludur. Başvuruda bulunan öğrencinin başvurusu sırasında genel not ortalamasının en az 3.00 olması ve anadal programının öğrenci başarı sıralamasında en üst %20 içinde bulunması gerekmektedir [Kaynak 2]. Bu 

### mezuniyet_005 (no_answer)

- Query: Mezuniyet için stajın tamamlanmış olması gerekir mi?
- Expected acceptable_source_terms: https://cdn.duzce.edu.tr/File/GetFile/2b51265c-b4f6-4ba4-b61d-7a8f4c6dd0ff, Lisans Eğitim Öğretim Yönetmeliği, Mezuniyet
- Returned sources: Yok
- Answer preview: Sayın öğrencimiz, Bölüm bazlı final sinavları için program her zaman belirlenen takvim ve düzenlemeler çerçevesinde oluşturulur. Ancak, dersin acildığı bölüm arası farklılıklar olabilir; bu nedenle duyuruların düzenli olarak kontrol edilmesi önemlidir [Kaynak 
- Chunk check: FOUND score=23 title=Diploma ve Mezuniyet Belgeleri Yönergesi url=https://cdn.duzce.edu.tr/File/GetFile/2b51265c-b4f6-4ba4-b61d-7a8f4c6dd0ff
- Chunk match reasons: expected_url, expected_term:Mezuniyet, query_terms:mezuniyet,olmasi,gerekir
- Chunk preview: MADDE 8 - (1) Diploma ekinde, mezuniyet tarihi, diploma numarası, alınan derecenin düzeyi, içeriği ve kullanım alanları, Üniversitenin eğitim-öğretim ve değerlendirme esasları ile ulusal eğitim sistemi hakkında bilgiler 

### mezuniyet_007 (wrong_source)

- Query: Bilgisayar mühendisliğinde mezun olabilmek için staj dışında hangi temel şartlar sağlanmalı?
- Expected acceptable_source_terms: https://panel.duzce.edu.tr//Dosya/GetFile/0bf859f9-969c-4171-b6ae-356555f8dd99, Lisans Eğitim Öğretim Yönetmeliği, Mezuniyet
- Returned sources: CAP ve Yandal (https://bm.mf.duzce.edu.tr/sayfa/fdcb/cift-anadal-ve-yandal); Düzce Üniversitesi Mevzuat Komisyonu (https://cdn.duzce.edu.tr/File/GetFile/7772123f-c7c5-4552-b300-7147262a5377)
- Answer preview: Sayın öğrencimiz, Genel not ortalaması şartları hakkında bilgi vermek için aşağıdaki noktaları göz önünde bulundurun: - Çift anadal programını tamamlamak isteyen öğrenci, mezuniyet not ortalamasına (anadal ve yandal) göre farklı derecelerde hak kazanabilir. [K

## yatay_gecis

- Fail count: 3
- Classes: no_answer=1, wrong_source=2
- Recommendation: Retrieval yanlis resmi belgeyi one cikariyor. Bu topic icin _candidate_score icinde kaynak basligi, URL oruntusu ve ilgili anahtar terimlere pozitif agirlik; sik karisan topic'lere negatif agirlik eklenmeli.

### yatay_gecis_003 (wrong_source)

- Query: Yatay geçiş için gerekli belgeler nelerdir?
- Expected acceptable_source_terms: https://cdn.duzce.edu.tr/File/GetFile/5f02abd3-26b9-499e-9719-bb7062f10337, Yatay Geçiş Başvuru Esasları, Yatay Geçiş
- Returned sources: Öğrenci Belgesi ve Transkript İşlemleri (https://cdn.duzce.edu.tr/File/GetFile/6ed94076-f5da-44ae-aa4a-244b58ad825e)
- Answer preview: Sayin ogrencimiz, Transkript/not durum belgesi oncelikle e-Devlet Kapisi uzerinden temin edilmelidir. E-Devlet uzerinden temin edilemeyen veya islak imzali/onayli gereken durumlarda ilgili birimin Ogrenci Isleri Burosuna basvurulabilir.  Dayanak: [Kaynak 1] Öğ

### yatay_gecis_004 (wrong_source)

- Query: Yatay geçiş başvuruları ne zaman açıklanır?
- Expected acceptable_source_terms: https://cdn.duzce.edu.tr/File/GetFile/cd218698-9e13-4d5b-911a-deeb096326de, Yatay Geçiş Başvuru Esasları, Yatay Geçiş
- Returned sources: AKREDİTASYON, AKADEMİK DEĞERLENDİRME (https://cdn.duzce.edu.tr/File/GetFile/7d4fa754-666a-46fb-8a45-97f53181c02f); 13 Haziran 2020 CUMARTESİ (https://cdn.duzce.edu.tr/File/GetFile/0e367049-d3e0-4b95-9a17-7a417d928fe0); 5D784293 4Abb 4Bc9 8A53 F15Dddaee6E9 (https://cdn.duzce.edu.tr/File/GetFile/5d784293-4abb-4bc9-8a53-f15dddaee6e9)
- Answer preview: Sayın öğrencimiz, Resmi kaynaklarda yer alan ilgili bilgi asagidadir: - Sonuçlar geri bildirimler doğrultusunda stratejik çerçeve sürekli güncellenerek, hedeﬂenen değişimin kalıcı ve etkili bir biçimde kurumsal kültüre entegre edilmesi sağlanmaktadır. - Ulusla

### yatay_gecis_006 (no_answer)

- Query: Yatay geçiş başvurusu online mı yapılıyor?
- Expected acceptable_source_terms: https://cdn.duzce.edu.tr/File/GetFile/5f02abd3-26b9-499e-9719-bb7062f10337, Yatay Geçiş Başvuru Esasları, Yatay Geçiş
- Returned sources: Yok
- Answer preview: Sayın öğrencimiz, Bu konuda resmi belgelerde bilgiye ulasilamadi. Lutfen Ogrenci Isleri birimi ile iletisime geciniz.
- Chunk check: FOUND score=23 title=DÜZCE ÜNİVERSİTESİ KURUM İÇİ VE KURUMLAR ARASI YATAY GEÇİŞ url=https://cdn.duzce.edu.tr/File/GetFile/5f02abd3-26b9-499e-9719-bb7062f10337
- Chunk match reasons: expected_url, expected_term:Yatay Geçiş, query_terms:yatay,gecis,basvurusu
- Chunk preview: (5) Üniversite bünyesindeki aynı düzeyde fakat farklı merkezi yerleştirme puan türü ile öğrenci kabul eden diploma programları arasında yatay geçiş başvurusu yapılabilmesi için, öğrencinin merkezi sınava girdiği yıl itib

## cap_yandal

- Fail count: 2
- Classes: no_answer=1, wrong_source=1
- Recommendation: No-answer vakalarinda chunks.json kanitini kontrol et. Kanit varsa esik, topic metadata veya skorlamayi guclendir; kanit yoksa crawler/kaynak listesi eksik.

### cap_yandal_002 (no_answer)

- Query: ÇAP için ortalama kaç olmalı?
- Expected acceptable_source_terms: https://bm.mf.duzce.edu.tr/sayfa/fdcb/cift-anadal-ve-yandal, ÇAP ve Yandal Başvuru Esasları, ÇAP / Yandal
- Returned sources: Yok
- Answer preview: Sayın öğrencimiz, Bu konuda resmi belgelerde bilgiye ulasilamadi. Lutfen Ogrenci Isleri birimi ile iletisime geciniz.
- Chunk check: FOUND score=21 title=CAP ve Yandal url=https://bm.mf.duzce.edu.tr/sayfa/fdcb/cift-anadal-ve-yandal
- Chunk match reasons: expected_url, expected_term:ÇAP / Yandal, query_terms:ortalama
- Chunk preview: Biyosistem Mühendisliği (Ziraat Fakültesi) Çift Anadal Öğrencisi Alacak Bölüm Adı Çift Anadal Programı Kontenjanı Çift Anadal Öğrencisi Alacak Bölüm Adı Çift Anadal Programı Bilgisayar Mühendisliği Bölümü Biyomedikal Müh

### cap_yandal_004 (wrong_source)

- Query: ÇAP başvuruları hangi tarihlerde yapılır?
- Expected acceptable_source_terms: https://bm.mf.duzce.edu.tr/sayfa/fdcb/cift-anadal-ve-yandal, ÇAP ve Yandal Başvuru Esasları, ÇAP / Yandal
- Returned sources: 5D784293 4Abb 4Bc9 8A53 F15Dddaee6E9 (https://cdn.duzce.edu.tr/File/GetFile/5d784293-4abb-4bc9-8a53-f15dddaee6e9)
- Answer preview: Sayın öğrencimiz, Resmi kaynaklarda yer alan ilgili bilgi asagidadir: - Uluslararası işbirliği ile gerçekleştirilecek projelerin duyuru, başvuru ve yürütme işlerinin sekretaryasını sağlamak. Koordinatörlüğün faaliyet alanları Kesin uygulama icin ilgili resmi k

## sinavlar

- Fail count: 2
- Classes: no_answer=0, wrong_source=2
- Recommendation: Retrieval yanlis resmi belgeyi one cikariyor. Bu topic icin _candidate_score icinde kaynak basligi, URL oruntusu ve ilgili anahtar terimlere pozitif agirlik; sik karisan topic'lere negatif agirlik eklenmeli.

### sinavlar_001 (wrong_source)

- Query: Mazeret sınavına kimler girebilir?
- Expected acceptable_source_terms: https://cdn.duzce.edu.tr/File/GetFile/0a10d86d-7915-4164-b60b-2e4d3ac59f19, Yaz Okulu Uygulama Esasları, Sınavlar
- Returned sources: Bu belge,güvenli elektronik imza ile imzalanmıştır. (https://cdn.duzce.edu.tr/File/GetFile/06193c82-1968-4c61-9891-d55538ef9528)
- Answer preview: Sayın öğrencimiz, Başvuru süresi içinde belgelerden herhangi biri eksik olan adayların başvuruları 3 gün içerisinde tamamlanması gerekmektedir [Kaynak 1]. Eğer sorunuz veya ihtiyacınız varsa, lütfen Öğrenci İşleri Daire Başkanlığı ile iletişime geçiniz. Size y

### sinavlar_003 (wrong_source)

- Query: Final tarihleri nereden öğrenilir?
- Expected acceptable_source_terms: https://cdn.duzce.edu.tr/File/GetFile/264a1c72-d5aa-4885-961e-75e0ba94acd5, 2024-2025 Eğitim-Öğretim Yılı Akademik Takvimi, Sınavlar
- Returned sources: Bu belge,güvenli elektronik imza ile imzalanmıştır. (https://cdn.duzce.edu.tr/File/GetFile/06193c82-1968-4c61-9891-d55538ef9528)
- Answer preview: Sayın öğrencimiz, Final sınavları için belirli tarihler genellikle akademik takvim ve ilgili birim/Oğrenci İşleri Daire Başkanlığı duyuruları üzerinden bilgilendirilir. Bu nedenle, final sınavlarının tarihleri hakkında güncel ve kesin bilgi alabilmek için her 
