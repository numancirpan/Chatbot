from __future__ import annotations

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
GOLDEN_PATH = ROOT_DIR / "data" / "golden_questions.json"


ADDITIONAL_CASES = {
    "staj": [
        {
            "base_case_id": "staj_002",
            "intent": "staj_sigorta",
            "query": "Staj sigortasını kim yapıyor?",
            "query_variants": [
                "Staj sigortamı üniversite mi yapıyor?",
                "zorunlu staj sigortasını kim karşılıyor",
                "staj sigorta işlemleri nasıl oluyor",
            ],
            "followups": [
                "Sigorta için ayrıca başvuru yapmam gerekir mi?",
                "Staj başlamadan önce sigorta tamamlanır mı?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["sigorta işlemleri", "sorumlu birim", "başvuru süreci"],
        },
        {
            "base_case_id": "staj_003",
            "intent": "staj_rapor_teslim",
            "query": "Staj raporunu ne zaman teslim etmeliyim?",
            "query_variants": [
                "staj defteri ne zaman teslim edilir",
                "staj raporu yükleme süresi ne kadar",
                "staj dosyasını geç teslim edersem ne olur",
            ],
            "followups": [
                "SBS üzerinden mi yükleniyor?",
                "Teslim süresi bölümden bölüme değişir mi?",
            ],
            "answer_style": "kurumsal_maddeli",
            "expected_points": ["teslim süresi", "SBS yükleme", "değerlendirme süreci"],
        },
        {
            "base_case_id": "staj_001",
            "intent": "staj_donemleri",
            "query": "Zorunlu staj hangi dönemlerde yapılır?",
            "query_variants": [
                "staj 1 ve staj 2 hangi dönemlerde yapılmalı",
                "zorunlu stajı hangi yarıyıllardan sonra yaparım",
                "bilgisayar mühendisliği staj dönemleri neler",
            ],
            "followups": [
                "Döneminde yapamazsam sonraki yaz yapabilir miyim?",
                "Staj dersini sonra seçerek saydırabilir miyim?",
            ],
            "answer_style": "kurumsal_maddeli",
            "expected_points": ["yarıyıl bilgisi", "staj dönemleri", "bölüm bazlı açıklama"],
        },
        {
            "base_case_id": "staj_006",
            "intent": "staj_doneminde_yapamazsa",
            "query": "Bilgisayar mühendisliği öğrencisi döneminde staj yapamazsa daha sonra nasıl saydırır?",
            "query_variants": [
                "stajımı döneminde yapamazsam sonra nasıl saydırırım",
                "bilgisayar mühendisliğinde stajı geciktirirsem ne yaparım",
                "döneminde yapılmayan staj daha sonra nasıl işlenir",
            ],
            "followups": [
                "Takip eden yarıyılda OBS'de ders almak gerekir mi?",
                "Dersi daha önce aldıysam tekrar almam gerekir mi?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["takip eden yarıyıl", "OBS staj dersi", "saydırma süreci"],
        },
    ],
    "ders_kaydi": [
        {
            "base_case_id": "ders_kaydi_001",
            "intent": "ders_kaydi_danisman",
            "query": "Ders kaydında danışman onayı gerekiyor mu?",
            "query_variants": [
                "ders seçiminden sonra danışman onayı var mı",
                "kayıt yenilemede danışman onaylamazsa ne olur",
                "ders kaydı danışman onayıyla mı tamamlanır",
            ],
            "followups": [
                "Onay gelmezse kayıt geçersiz olur mu?",
                "OBS üzerinden takip edebilir miyim?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["danışman onayı", "OBS süreci", "kayıt tamamlanması"],
        },
        {
            "base_case_id": "ders_kaydi_002",
            "intent": "ders_kaydi_mazeret",
            "query": "Mazeretli ders kaydı ne zaman yapılır?",
            "query_variants": [
                "mazeretli kayıt tarihleri ne zaman",
                "ders kaydını kaçırırsam mazeretli kayıt var mı",
                "geç kalanlar için ders kaydı ne zaman açılır",
            ],
            "followups": [
                "Mazeretli kayıt akademik takvimde yazar mı?",
                "Her öğrenci mazeretli kayıt yapabilir mi?",
            ],
            "answer_style": "kurumsal_maddeli",
            "expected_points": ["mazeretli kayıt tarihi", "akademik takvim", "başvuru koşulu"],
        },
        {
            "base_case_id": "ders_kaydi_003",
            "intent": "ders_kaydi_alttan_ustten",
            "query": "Alttan dersim varken üstten ders alabilir miyim?",
            "query_variants": [
                "üstten ders alma şartları neler",
                "alttan ders olunca ekstra ders seçilir mi",
                "bir dönemde üstten ders seçmek mümkün mü",
            ],
            "followups": [
                "Kredi sınırı buna göre değişir mi?",
                "Danışman onayı gerekir mi?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["üstten ders alma", "kredi sınırı", "koşullar"],
        },
    ],
    "add_drop": [
        {
            "base_case_id": "add_drop_001",
            "intent": "add_drop_son_gun",
            "query": "Add-drop işlemleri hangi gün sona erer?",
            "query_variants": [
                "add-drop son gün ne zaman",
                "ders ekle sil haftası ne zaman biter",
                "add drop son tarihi nedir",
            ],
            "followups": [
                "Takvimde lisans için ayrı mı gösteriliyor?",
                "Son gün saat sınırı olur mu?",
            ],
            "answer_style": "kurumsal_maddeli",
            "expected_points": ["bitiş tarihi", "akademik takvim", "add-drop dönemi"],
        },
        {
            "base_case_id": "add_drop_002",
            "intent": "add_drop_danisman",
            "query": "Add-drop sırasında danışman onayı gerekiyor mu?",
            "query_variants": [
                "ders ekle sil işlemi danışman onaylı mı",
                "add-drop yapınca danışman onayı şart mı",
                "ders değişikliğinde danışman onayı gerekir mi",
            ],
            "followups": [
                "OBS’de değişiklik hemen görünür mü?",
                "Onay olmadan ders ekleme tamamlanır mı?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["danışman onayı", "ders ekleme/silme", "OBS süreci"],
        },
        {
            "base_case_id": "add_drop_003",
            "intent": "add_drop_kredi",
            "query": "Add-drop döneminde kredi sınırını aşabilir miyim?",
            "query_variants": [
                "add-drop yaparken maksimum kredi değişir mi",
                "ders ekle sil haftasında kredi limiti aşılır mı",
                "kredi sınırı add-drop’ta da geçerli mi",
            ],
            "followups": [
                "Danışman onayıyla aşılabilir mi?",
                "Üstten ders alırken durum değişir mi?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["kredi sınırı", "add-drop kuralı", "koşullar"],
        },
    ],
    "devamsizlik": [
        {
            "base_case_id": "devamsizlik_001",
            "intent": "devamsizlik_teori_uygulama",
            "query": "Teorik ve uygulamalı derslerde devamsızlık sınırı aynı mı?",
            "query_variants": [
                "uygulamalı ders devamsızlığı farklı mı",
                "teori ve uygulamada devam şartı değişir mi",
                "laboratuvar derslerinde devamsızlık nasıl olur",
            ],
            "followups": [
                "Yüzde olarak mı hesaplanır?",
                "Ders planında ayrıca belirtilir mi?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["teorik ders", "uygulamalı ders", "devam zorunluluğu"],
        },
        {
            "base_case_id": "devamsizlik_002",
            "intent": "devamsizlik_final",
            "query": "Devamsızlıktan kalan öğrenci finale girebilir mi?",
            "query_variants": [
                "devamsızlıktan kalınca final hakkı olur mu",
                "devamsızlık nedeniyle sınava girilir mi",
                "devamsız öğrencinin dönem sonu sınav hakkı var mı",
            ],
            "followups": [
                "Bütünleme hakkı da gider mi?",
                "OBS’de devamsız görünür mü?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["final hakkı", "devamsızlıktan kalma", "sınav durumu"],
        },
        {
            "base_case_id": "devamsizlik_001",
            "intent": "devamsizlik_rapor",
            "query": "Sağlık raporu devamsızlığı siler mi?",
            "query_variants": [
                "rapor alınca devamsızlık düşer mi",
                "hastane raporu devamsızlık yerine geçer mi",
                "mazeret raporu devam şartını kaldırır mı",
            ],
            "followups": [
                "Rapor mazeret sınavı için geçerli olur mu?",
                "Derse devam zorunluluğu yine de sürer mi?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["sağlık raporu", "devam zorunluluğu", "mazeret ayrımı"],
        },
    ],
    "sinavlar": [
        {
            "base_case_id": "sinavlar_001",
            "intent": "sinavlar_mazeret_basvuru",
            "query": "Mazeret sınavı için nasıl başvuru yapılır?",
            "query_variants": [
                "mazeret sınav başvurusu nasıl oluyor",
                "mazeret sınavına girmek için ne yapmalıyım",
                "sınav mazereti nereye bildirilir",
            ],
            "followups": [
                "Rapor yeterli olur mu?",
                "Başvuru süresi kaç gündür?",
            ],
            "answer_style": "kurumsal_maddeli",
            "expected_points": ["başvuru süreci", "mazeret belgesi", "süre"],
        },
        {
            "base_case_id": "sinavlar_002",
            "intent": "sinavlar_but_final",
            "query": "Bütünleme notu final notunun yerine geçer mi?",
            "query_variants": [
                "büt notu finalin yerine yazılır mı",
                "bütünleme sınavı finali siler mi",
                "final ve bütünleme notundan hangisi geçerli olur",
            ],
            "followups": [
                "Daha düşük alırsam son not değişir mi?",
                "Her ders için bütünleme var mı?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["bütünleme notu", "final notu", "geçerli not"],
        },
        {
            "base_case_id": "sinavlar_003",
            "intent": "sinavlar_yaz_okulu_but",
            "query": "Yaz okulunda bütünleme sınavı olur mu?",
            "query_variants": [
                "yaz okulunda büt var mı",
                "yaz okulu derslerinde bütünleme yapılıyor mu",
                "yaz okulunda finalden sonra ek sınav olur mu",
            ],
            "followups": [
                "Yaz okulu uygulama esaslarında yazıyor mu?",
                "Mazeret sınavı ile karışır mı?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["yaz okulu", "bütünleme sınavı", "uygulama esasları"],
        },
    ],
    "not_sistemi_ortalama": [
        {
            "base_case_id": "not_sistemi_ortalama_001",
            "intent": "not_sistemi_agno",
            "query": "AGNO ve GANO aynı şey mi?",
            "query_variants": [
                "agno ile gano farkı nedir",
                "genel not ortalaması ile agno aynı mı",
                "ortalama türleri nelerdir",
            ],
            "followups": [
                "Mezuniyet için hangisi kullanılır?",
                "Dönem ortalaması ayrı mı hesaplanır?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["AGNO/GANO", "ortalama türü", "hesaplama mantığı"],
        },
        {
            "base_case_id": "not_sistemi_ortalama_002",
            "intent": "not_sistemi_kosullu_gecis",
            "query": "DD ve DC notlarıyla koşullu geçiş nasıl oluyor?",
            "query_variants": [
                "dc dd ile ders geçilir mi",
                "koşullu geçme için ortalama kaç olmalı",
                "dd dc notu başarısız sayılır mı",
            ],
            "followups": [
                "AGNO düşükse kalır mıyım?",
                "Tekrar almam gerekir mi?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["DD/DC", "koşullu geçiş", "ortalama koşulu"],
        },
        {
            "base_case_id": "not_sistemi_ortalama_003",
            "intent": "not_sistemi_tekrar",
            "query": "Not yükseltmek için başarılı olduğum dersi tekrar alabilir miyim?",
            "query_variants": [
                "geçtiğim dersi not yükseltmek için tekrar seçebilir miyim",
                "başarılı dersi ortalama artırmak için almak mümkün mü",
                "harf notunu yükseltmek için ders tekrarı yapılır mı",
            ],
            "followups": [
                "Son alınan not mu geçerli olur?",
                "Kredi hesabı değişir mi?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["ders tekrarı", "not yükseltme", "geçerli not"],
        },
    ],
    "mezuniyet": [
        {
            "base_case_id": "mezuniyet_001",
            "intent": "mezuniyet_eksik_ders",
            "query": "Bir dersim eksikken mezun olabilir miyim?",
            "query_variants": [
                "tek ders kalınca mezun sayılır mıyım",
                "mezuniyet için tüm dersleri geçmiş olmak zorunlu mu",
                "eksik dersle mezuniyet olur mu",
            ],
            "followups": [
                "Tek ders sınavı varsa durum değişir mi?",
                "Staj da mezuniyet şartına dahil mi?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["ders tamamlama", "mezuniyet şartı", "tek ders durumu"],
        },
        {
            "base_case_id": "mezuniyet_002",
            "intent": "mezuniyet_staj",
            "query": "Mezuniyet için stajın tamamlanmış olması gerekir mi?",
            "query_variants": [
                "stajı yapmadan mezun olunur mu",
                "mezuniyet için zorunlu staj şart mı",
                "staj eksikse diploma alınır mı",
            ],
            "followups": [
                "Tek ders sınavı olsa bile staj zorunlu mu?",
                "Bölüm bazlı farklılık olabilir mi?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["staj şartı", "mezuniyet koşulu", "tamamlama zorunluluğu"],
        },
        {
            "base_case_id": "mezuniyet_003",
            "intent": "mezuniyet_diploma",
            "query": "Diplomamı mezun olduktan ne kadar sonra alabilirim?",
            "query_variants": [
                "mezuniyet sonrası diploma ne zaman verilir",
                "diploma teslim süreci nasıl işler",
                "geçici mezuniyet belgesi alınabilir mi",
            ],
            "followups": [
                "Diploma için ayrıca başvuru gerekir mi?",
                "Geçici mezuniyet belgesi önce verilir mi?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["diploma süreci", "teslim", "geçici mezuniyet belgesi"],
        },
        {
            "base_case_id": "mezuniyet_001",
            "intent": "mezuniyet_temin_sartlar",
            "query": "Bilgisayar mühendisliğinde mezun olabilmek için staj dışında hangi temel şartlar sağlanmalı?",
            "query_variants": [
                "mezuniyet için staj dışında hangi koşullar var",
                "bilgisayar mühendisliğinde diploma almak için temel mezuniyet şartları neler",
                "staj hariç mezuniyet koşulları nelerdir",
            ],
            "followups": [
                "Derslerin tamamı başarılı olmalı mı?",
                "Genel not ortalaması şartı da aranır mı?",
            ],
            "answer_style": "kurumsal_maddeli",
            "expected_points": ["mezuniyet şartları", "derslerin tamamlanması", "ilgili mevzuat koşulları"],
        },
    ],
    "cap_yandal": [
        {
            "base_case_id": "cap_yandal_001",
            "intent": "cap_yandal_takvim",
            "query": "ÇAP başvuruları hangi tarihlerde yapılır?",
            "query_variants": [
                "çift anadal başvuru tarihleri ne zaman",
                "çap takvimi nerede açıklanır",
                "çap başvuru dönemi hangi haftada olur",
            ],
            "followups": [
                "Her dönem açılır mı?",
                "Duyurular bölüm sayfasında mı yayımlanır?",
            ],
            "answer_style": "kurumsal_maddeli",
            "expected_points": ["başvuru tarihi", "duyuru", "çap takvimi"],
        },
        {
            "base_case_id": "cap_yandal_002",
            "intent": "cap_yandal_ayni_anda",
            "query": "Aynı anda hem ÇAP hem yandal yapılabilir mi?",
            "query_variants": [
                "çap ve yandal birlikte okunur mu",
                "aynı dönemde hem çift anadal hem yandal olur mu",
                "iki program birden yürütülebilir mi",
            ],
            "followups": [
                "Birini bırakınca diğeri devam eder mi?",
                "Ortalama şartı ikisi için de geçerli mi?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["çap", "yandal", "birlikte yürütme"],
        },
        {
            "base_case_id": "cap_yandal_003",
            "intent": "cap_yandal_birakma",
            "query": "Yandalı bırakmak istersem ne olur?",
            "query_variants": [
                "yandal programından ayrılabilir miyim",
                "yandalı silince ana program etkilenir mi",
                "yandal bırakma işlemi nasıl yapılır",
            ],
            "followups": [
                "ÇAP için de benzer kural var mı?",
                "Başvuru ile mi bırakılır?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["programdan ayrılma", "başvuru", "ana program etkisi"],
        },
    ],
    "yatay_gecis": [
        {
            "base_case_id": "yatay_gecis_001",
            "intent": "yatay_gecis_takvim",
            "query": "Yatay geçiş başvuruları ne zaman açıklanır?",
            "query_variants": [
                "yatay geçiş takvimi ne zaman yayımlanır",
                "başvuru tarihleri hangi dönemde duyurulur",
                "yatay geçiş tarihleri nerede ilan edilir",
            ],
            "followups": [
                "Her yıl değişir mi?",
                "Sonuçlar aynı yerde mi duyurulur?",
            ],
            "answer_style": "kurumsal_maddeli",
            "expected_points": ["başvuru takvimi", "duyuru", "ilan"],
        },
        {
            "base_case_id": "yatay_gecis_002",
            "intent": "yatay_gecis_hazirlik",
            "query": "Hazırlık okuyan öğrenci yatay geçiş yapabilir mi?",
            "query_variants": [
                "hazırlık sınıfındayken yatay geçiş olur mu",
                "hazırlık öğrencisi kurum içi geçiş yapabilir mi",
                "hazırlıkta okurken bölüm değiştirilebilir mi",
            ],
            "followups": [
                "Merkezi puanla durum değişir mi?",
                "Dil hazırlığı şartı etkiler mi?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["hazırlık sınıfı", "yatay geçiş koşulu", "başvuru durumu"],
        },
        {
            "base_case_id": "yatay_gecis_003",
            "intent": "yatay_gecis_online",
            "query": "Yatay geçiş başvurusu online mı yapılıyor?",
            "query_variants": [
                "yatay geçiş başvurusu internetten mi",
                "evraklar online yükleniyor mu",
                "başvuru sistemi var mı",
            ],
            "followups": [
                "Belgeleri ayrıca teslim etmek gerekir mi?",
                "Sonuçlar online açıklanır mı?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["online başvuru", "evrak yükleme", "başvuru sistemi"],
        },
    ],
    "harc_ucret": [
        {
            "base_case_id": "harc_ucret_001",
            "intent": "harc_ucret_son_gun",
            "query": "Harç ödeme için son gün ne zaman?",
            "query_variants": [
                "katkı payı son ödeme tarihi nedir",
                "öğrenim ücreti ödeme son günü ne zaman",
                "harç yatırma tarihi ne zaman bitiyor",
            ],
            "followups": [
                "Akademik takvimde ayrıca gösterilir mi?",
                "Gecikince kayıt yenileme olur mu?",
            ],
            "answer_style": "kurumsal_maddeli",
            "expected_points": ["son ödeme tarihi", "akademik takvim", "kayıt yenileme ilişkisi"],
        },
        {
            "base_case_id": "harc_ucret_002",
            "intent": "harc_ucret_iade",
            "query": "Fazla yatırılan harç iade edilir mi?",
            "query_variants": [
                "yanlış yatırılan katkı payı geri alınır mı",
                "fazla ödeme olursa iade var mı",
                "öğrenim ücreti iadesi nasıl yapılır",
            ],
            "followups": [
                "İade için dilekçe gerekir mi?",
                "Öğrenci işleri mi muhasebe mi ilgilenir?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["iade süreci", "dilekçe", "ilgili birim"],
        },
        {
            "base_case_id": "harc_ucret_003",
            "intent": "harc_ucret_kayit_engel",
            "query": "Harç ödenmezse öğrenci ders seçebilir mi?",
            "query_variants": [
                "katkı payı yatırmadan obs açılır mı",
                "ücret ödenmeden kayıt tamamlanır mı",
                "harç borcu varsa ders seçimi olur mu",
            ],
            "followups": [
                "Danışman onayı da bekler mi?",
                "Mazeretli kayıt için durum değişir mi?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["ödeme zorunluluğu", "ders seçimi", "kayıt engeli"],
        },
    ],
    "burs": [
        {
            "base_case_id": "burs_001",
            "intent": "burs_belge",
            "query": "Burs başvurusunda hangi belgeler istenir?",
            "query_variants": [
                "burs için gerekli evraklar neler",
                "burs başvurusu belge listesi nedir",
                "hangi belgelerle burs başvurusu yapacağım",
            ],
            "followups": [
                "Belgeler online mi yüklenir?",
                "Eksik belge olursa başvuru reddedilir mi?",
            ],
            "answer_style": "kurumsal_maddeli",
            "expected_points": ["gerekli belgeler", "başvuru evrakı", "yükleme/teslim"],
        },
        {
            "base_case_id": "burs_002",
            "intent": "burs_sonuclar",
            "query": "Burs sonuçları ne zaman açıklanır?",
            "query_variants": [
                "burs başvuru sonuçları ne zaman belli olur",
                "burs kazananlar ne zaman ilan edilir",
                "sonuç duyurusu nereden yapılır",
            ],
            "followups": [
                "Duyuru öğrenci işleri sayfasında mı olur?",
                "Sonuçlar bireysel olarak bildirilir mi?",
            ],
            "answer_style": "kurumsal_maddeli",
            "expected_points": ["sonuç tarihi", "duyuru", "ilan yöntemi"],
        },
        {
            "base_case_id": "burs_003",
            "intent": "burs_basarisizlik",
            "query": "Başarısız olursam bursum kesilir mi?",
            "query_variants": [
                "ortalama düşünce burs kesilir mi",
                "akademik başarısızlık bursu etkiler mi",
                "burs devamı için başarı şartı var mı",
            ],
            "followups": [
                "Disiplin cezası ile aynı şey mi?",
                "Her burs türünde kural aynı mı?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["başarı şartı", "burs kesilmesi", "burs türü farkı"],
        },
    ],
    "ogrenci_belgesi_transkript": [
        {
            "base_case_id": "ogrenci_belgesi_transkript_001",
            "intent": "ogrenci_belgesi_e_devlet",
            "query": "Öğrenci belgesini e-Devlet üzerinden alabilir miyim?",
            "query_variants": [
                "öğrenci belgesi e devletten çıkar mı",
                "okula gitmeden öğrenci belgesi alınır mı",
                "online öğrenci belgesi mümkün mü",
            ],
            "followups": [
                "Islak imzalı istenirse ne yapmalıyım?",
                "Barkodlu belge geçerli olur mu?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["e-Devlet", "öğrenci belgesi", "barkod/ıslak imza"],
        },
        {
            "base_case_id": "ogrenci_belgesi_transkript_003",
            "intent": "transkript_ucret",
            "query": "İngilizce transkript ücretli mi?",
            "query_variants": [
                "yabancı dilde transkript için ücret ödenir mi",
                "ingilizce transkript paralı mı",
                "transkript alırken ücret istenir mi",
            ],
            "followups": [
                "Kaç günde hazırlanır?",
                "Online talep edilebilir mi?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["İngilizce transkript", "ücret", "talep süreci"],
        },
        {
            "base_case_id": "ogrenci_belgesi_transkript_002",
            "intent": "transkript_onayli",
            "query": "Onaylı transkripti nereden alırım?",
            "query_variants": [
                "imzalı kaşeli transkript nasıl alınır",
                "resmi transkript için nereye başvurulur",
                "onaylı not dökümü öğrenci işlerinden mi alınır",
            ],
            "followups": [
                "E-devlet transkripti yeterli olur mu?",
                "İngilizce onaylı transkript de veriliyor mu?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["onaylı transkript", "başvuru birimi", "belge türü"],
        },
    ],
    "askerlik_tecili": [
        {
            "base_case_id": "askerlik_tecili_001",
            "intent": "askerlik_tecili_sure",
            "query": "Öğrencilik nedeniyle tecil ne zamana kadar sürer?",
            "query_variants": [
                "öğrenci tecili kaç yaşına kadar geçerli",
                "askerlik tecil süresi ne kadar",
                "öğrenciyken tecil bitişi nasıl hesaplanır",
            ],
            "followups": [
                "Program süresine göre değişir mi?",
                "Kayıt dondurunca etkilenir mi?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["tecil süresi", "öğrencilik durumu", "bitiş"],
        },
        {
            "base_case_id": "askerlik_tecili_002",
            "intent": "askerlik_mezuniyet",
            "query": "Mezun olunca askerlik tecilim hemen bozulur mu?",
            "query_variants": [
                "mezuniyet sonrası tecil devam eder mi",
                "diploma alınca askerlik ertelenmesi biter mi",
                "mezun olduktan sonra askerlik süreci nasıl işler",
            ],
            "followups": [
                "Ek tecil süresi olur mu?",
                "Askerlik şubesi mi belirler?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["mezuniyet sonrası", "tecil durumu", "askerlik şubesi"],
        },
        {
            "base_case_id": "askerlik_tecili_003",
            "intent": "askerlik_ilisik",
            "query": "Kaydım silinirse askerlik tecilim ne olur?",
            "query_variants": [
                "ilişik kesilince tecil bozulur mu",
                "üniversite kaydı kapanırsa askerlik ertelenmesi devam eder mi",
                "öğrencilik bitince tecil iptal olur mu",
            ],
            "followups": [
                "Kayıt dondurma ile aynı şey mi?",
                "Askerlik şubesine ayrıca bildirilir mi?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["ilişik kesme", "tecil iptali", "bildirim"],
        },
    ],
    "disiplin_islemleri": [
        {
            "base_case_id": "disiplin_islemleri_001",
            "intent": "disiplin_kinama",
            "query": "Kınama cezası hangi fiiller için verilir?",
            "query_variants": [
                "uyarma ve kınama cezası farkı nedir",
                "kınama cezası gerektiren durumlar neler",
                "hangi davranışlar disiplin cezasına girer",
            ],
            "followups": [
                "Yönetmelikte madde madde yazıyor mu?",
                "Daha ağır cezalar da var mı?",
            ],
            "answer_style": "kurumsal_maddeli",
            "expected_points": ["kınama cezası", "fiiller", "disiplin türleri"],
        },
        {
            "base_case_id": "disiplin_islemleri_002",
            "intent": "disiplin_savunma",
            "query": "Savunma alınmadan disiplin cezası verilebilir mi?",
            "query_variants": [
                "disiplin sürecinde savunma zorunlu mu",
                "öğrenciden savunma istenmeden ceza çıkar mı",
                "savunma hakkı olmadan işlem yapılır mı",
            ],
            "followups": [
                "Savunma için süre verilir mi?",
                "Yazılı savunma gerekir mi?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["savunma hakkı", "süreç", "öğrenci hakkı"],
        },
        {
            "base_case_id": "disiplin_islemleri_002",
            "intent": "disiplin_itiraz_sure",
            "query": "Disiplin kararına kaç gün içinde itiraz edilir?",
            "query_variants": [
                "disiplin cezası itiraz süresi nedir",
                "karara ne kadar sürede başvurulur",
                "disiplin kararına karşı süre kaç gün",
            ],
            "followups": [
                "İtiraz nereye yapılır?",
                "Dava açma süresi farklı mı?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["itiraz süresi", "başvuru mercii", "disiplin kararı"],
        },
    ],
    "akademik_takvim_duyurular": [
        {
            "base_case_id": "akademik_takvim_duyurular_001",
            "intent": "akademik_takvim_kayit",
            "query": "Kayıt yenileme tarihleri akademik takvimde yazar mı?",
            "query_variants": [
                "ders kaydı tarihleri akademik takvimde olur mu",
                "kayıt yenileme günleri nerede açıklanır",
                "akademik takvimde ders kayıt tarihi var mı",
            ],
            "followups": [
                "Mazeretli kayıt da eklenir mi?",
                "Her fakülte için aynı mı olur?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["kayıt yenileme", "akademik takvim", "mazeretli kayıt"],
        },
        {
            "base_case_id": "akademik_takvim_duyurular_002",
            "intent": "duyurular_staj",
            "query": "Öğrenci işleri duyurularını düzenli olarak nereden takip etmeliyim?",
            "query_variants": [
                "güncel öğrenci işleri duyuruları nerede olur",
                "duyurular için hangi sayfayı izlemeliyim",
                "resmi açıklamalar hangi sitede yayımlanıyor",
            ],
            "followups": [
                "Akademik takvim ve duyuru sayfası aynı mı?",
                "Bölüm duyuruları ayrıca kontrol edilmeli mi?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["duyuru sayfası", "resmi takip", "bölüm duyuruları"],
        },
        {
            "base_case_id": "akademik_takvim_duyurular_003",
            "intent": "akademik_takvim_sinav",
            "query": "Final ve bütünleme tarihleri akademik takvimde yer alır mı?",
            "query_variants": [
                "sınav tarihleri akademik takvimde olur mu",
                "final haftası nereden öğrenilir",
                "büt sınavları takvimde gösterilir mi",
            ],
            "followups": [
                "Yaz okulu sınavları da eklenir mi?",
                "Bölüm bazlı değişiklik olabilir mi?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["sınav tarihleri", "akademik takvim", "final/bütünleme"],
        },
    ],
    "yaz_okulu": [
        {
            "base_case_id": "yaz_okulu_001",
            "intent": "yaz_okulu_devam",
            "query": "Yaz okulunda devam zorunluluğu var mı?",
            "query_variants": [
                "yaz okulunda derse devam şart mı",
                "yaz okulu yoklama zorunlu mu",
                "yaz okulunda devamsızlık nasıl işler",
            ],
            "followups": [
                "Normal dönemden farklı mı?",
                "Uygulamalı derslerde durum değişir mi?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["devam zorunluluğu", "yaz okulu", "uygulama"],
        },
        {
            "base_case_id": "yaz_okulu_002",
            "intent": "yaz_okulu_ders_sayisi",
            "query": "Yaz okulunda en fazla kaç AKTS alınabilir?",
            "query_variants": [
                "yaz okulu kredi sınırı nedir",
                "yaz okulunda maksimum akts kaç",
                "kaç ders yerine kaç akts alınır",
            ],
            "followups": [
                "Ders sayısı ile AKTS sınırı aynı şey mi?",
                "Başka üniversiteden alınırsa değişir mi?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["AKTS sınırı", "ders sayısı", "yaz okulu limiti"],
        },
        {
            "base_case_id": "yaz_okulu_003",
            "intent": "yaz_okulu_dis_universite",
            "query": "Yaz okulunda başka üniversiteden ders alabilir miyim?",
            "query_variants": [
                "farklı üniversiteden yaz okulu dersi saydırılır mı",
                "başka okuldan yaz dersi almak mümkün mü",
                "misafir öğrenci olarak yaz okulu olur mu",
            ],
            "followups": [
                "Önceden onay almak gerekir mi?",
                "Muafiyet/intibak işlemi gerekir mi?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["başka üniversite", "ön onay", "saydırma/intibak"],
        },
        {
            "base_case_id": "yaz_okulu_001",
            "intent": "yaz_okulu_devam_bm",
            "query": "Bilgisayar mühendisliği yaz okulunda devam zorunluluğu var mı?",
            "query_variants": [
                "bilgisayar mühendisliğinde yaz okulunda devam zorunlu mu",
                "yaz okulunda devamsızlık kuralı var mı",
                "yaz okulu derslerinde yoklama zorunlu mu",
            ],
            "followups": [
                "Daha önce devam şartını sağladığım dersi tekrar alırsam durum değişir mi?",
                "Başka üniversiteden alınan yaz okulu dersinde devam şartı nasıl olur?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["yaz okulu", "devam zorunluluğu", "istisna koşulu"],
        },
        {
            "base_case_id": "yaz_okulu_006",
            "intent": "yaz_okulu_dis_universite_bm",
            "query": "Bilgisayar mühendisliği bölümünde yaz okulunda başka üniversiteden ders alabilir miyim?",
            "query_variants": [
                "bilgisayar mühendisliği öğrencisi yaz okulunda başka üniversiteden ders alabilir mi",
                "yaz okulunda başka üniversite dersi saydırılır mı",
                "misafir öğrenci gibi başka üniversiteden yaz dersi alabilir miyim",
            ],
            "followups": [
                "Bölüm başkanlığının onayı gerekir mi?",
                "Eşdeğerlik için AKTS ve içerik şartı aranır mı?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["başka üniversite", "bölüm onayı", "eşdeğerlik koşulu"],
        },
    ],
    "muafiyet_intibak": [
        {
            "base_case_id": "muafiyet_intibak_001",
            "intent": "muafiyet_tarih",
            "query": "Muafiyet başvurusu hangi tarihlerde yapılır?",
            "query_variants": [
                "muafiyet için son başvuru tarihi nedir",
                "ders saydırma başvurusu ne zaman yapılır",
                "intibak başvurusu hangi hafta alınır",
            ],
            "followups": [
                "Kayıt olduktan hemen sonra mı yapılır?",
                "Geç başvuru kabul edilir mi?",
            ],
            "answer_style": "kurumsal_maddeli",
            "expected_points": ["başvuru tarihi", "muafiyet dönemi", "geç başvuru"],
        },
        {
            "base_case_id": "muafiyet_intibak_002",
            "intent": "muafiyet_sonuc",
            "query": "İntibak sonucu nereden öğrenilir?",
            "query_variants": [
                "muafiyet sonucu nasıl açıklanır",
                "ders saydırma sonucu obs’de görünür mü",
                "intibak kararını nereden takip ederim",
            ],
            "followups": [
                "Bölüm kurul kararı gerekir mi?",
                "Sonuç öğrenci işleri tarafından mı duyurulur?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["intibak sonucu", "duyuru/takip", "kurul kararı"],
        },
        {
            "base_case_id": "muafiyet_intibak_003",
            "intent": "muafiyet_akts",
            "query": "Daha önce aldığım dersin kredisi düşükse yine de sayılır mı?",
            "query_variants": [
                "muafiyet için akts eşit olmak zorunda mı",
                "kredi farkı varsa ders saydırılır mı",
                "eşdeğer dersin kredisi farklıysa ne olur",
            ],
            "followups": [
                "İçerik benzerliği mi esas alınır?",
                "Kurul kararına mı bağlıdır?",
            ],
            "answer_style": "kurumsal_kisa",
            "expected_points": ["AKTS/kredi", "eşdeğerlik", "kurul değerlendirmesi"],
        },
    ],
}


def load_cases():
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def build_case_index(cases):
    return {case["id"]: case for case in cases}


def next_case_id(topic: str, sequence: int) -> str:
    return f"{topic}_{sequence:03d}"


def main() -> int:
    cases = load_cases()
    case_index = build_case_index(cases)
    position_by_id = {case["id"]: index for index, case in enumerate(cases)}

    appended = []
    updated = []
    for topic, additions in ADDITIONAL_CASES.items():
        for offset, addition in enumerate(additions, start=4):
            case_id = next_case_id(topic, offset)
            base = case_index[addition["base_case_id"]]
            new_case = {
                "id": case_id,
                "intent": addition["intent"],
                "topic": topic,
                "topic_label": base["topic_label"],
                "source_url": base["source_url"],
                "source_title": base["source_title"],
                "query": addition["query"],
                "query_variants": addition["query_variants"],
                "followups": addition["followups"],
                "answer_style": addition["answer_style"],
                "expected_points": addition["expected_points"],
            }
            if case_id in position_by_id:
                cases[position_by_id[case_id]] = new_case
                updated.append(case_id)
            else:
                position_by_id[case_id] = len(cases)
                cases.append(new_case)
                appended.append(case_id)

    GOLDEN_PATH.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"golden_cases": len(cases), "appended": len(appended), "updated": len(updated)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
