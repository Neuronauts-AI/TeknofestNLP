# Neuronauts AI - Turkce Klinik NLP Platformu

Bu repo, Turkce radyoloji rapor akisini iyilestirmeye odaklanan bir klinik NLP prototipini icerir. Mevcut calisma, metinden rapor uretimi, rapor uygunluk degerlendirmesi, canli uyari cikarma ve sesli dikte denemeleri etrafinda sekillenmistir.

Proje, stratejik olarak 3 modullu bir platform olarak dusunulmektedir:

1. Modül 1: Rapor uretim ve sesli dikte akisi
2. Modül 2: Otomatik kontrol, skorlama ve semantik arama
3. Modül 3: Doktor cumle tamamlama asistani

Bu repodaki aktif gelistirme su anda agirlikli olarak Modül 1 uzerindedir.

## Lisans

Bu repo Apache License 2.0 ile lisanslanmistir. Ayrintilar icin [LICENSE](LICENSE) dosyasina bakin.

## Simdiye Kadar Yapilanlar

### Veri Hazirligi

- MIMIC-CXR kaynakli rapor verileri cekildi.
- Metin odakli CSV ciktilari uretildi.
- Turkce rapor verisi olusturuldu.
- Veri klasorleri `app/`, `data/`, `docs/`, `scripts/`, `experiments/` olarak duzenlendi.

### Modül 1 - Rapor Uretim Demo Katmani

- FastAPI tabanli backend kuruldu.
- Tarayici uzerinden kullanilan demo arayuzu eklendi.
- Kullanici `Bulgular` ve `Sonuc` alanlarini doldurup rapor uretebiliyor.
- Ornek veri yukleme, rapor gecmisi ve canli akis iskeleti hazirlandi.
- `Örnekleri Yenile` akisi veri setinden rastgele ornek getirecek sekilde guncellendi.

### Rapor Uygunluk / Gerceklik Degerlendirmesi

- Hastalik siniflandirma yaklasimi projeden cikarildi.
- Bunun yerine, raporun gercek bir Turkce radyoloji raporu gibi gorunup gorunmedigini degerlendiren Ollama tabanli bir siniflandirma katmani kuruldu.
- Referans radyoloji raporu uslubu metin dosyalariyla projede tutuluyor.
- Promptlar ayri dosyalara ayrildi ve duzenlenebilir hale getirildi.

### Canli Uyari Sistemi

- Rapor metni uzerinden LLM tabanli canli kritik bulgu/uyari cikarma katmani eklendi.
- Uyarilar UI icinde gosteriliyor.
- Durum etiketleri Turkcelestirildi: `Mevcut`, `Belirsiz`, `Yok`.
- Uyari paneline filtreleme eklendi.

### Sesli Giris ve ASR

- Whisper tabanli ASR akisi eklendi.
- Mikrofon kaydi ve ses dosyasi yukleme ile transkript alma ozelligi kuruldu.
- Varsayilan ASR modeli `openai/whisper-large-v3` olarak ayarlandi.
- Sesli akista LLM ile transcript yeniden yazimi veya duzeltme yapilmiyor.
- Kullanici konusurken `Bulgular` ve `Sonuc` basliklarini soylediginde metin bu basliklara gore ayriliyor.
- Kotu kalite ses uzerinde ilk dayanıklılık testleri yapildi; ana darboğazın halen ASR kalitesi oldugu goruldu.

### Deneysel Bilesenler

- Ollama ile rapor uslubu siniflandirmasi icin ayri deneysel script eklendi.
- ASR dayaniklilik testi icin benchmark scripti eklendi.

## Mevcut Durum

Proje su anda calisan bir MVP/prototip seviyesindedir.

Uctan uca mevcut akis:

1. Kullanici metin girer veya sesli dikte yapar
2. Sistem `Bulgular` ve `Sonuc` alanlarini doldurur
3. Rapor olusturulur
4. Raporun Turkce radyoloji uslubuna uygunlugu degerlendirilir
5. Canli kritik uyari sinyalleri uretilebilir

Mevcut temel sinir:

- Dusuk kaliteli veya anlasilmasi zor seste medikal Turkce terimlerde bozulmalar gorulebiliyor.
- Bu durum sonraki rapor uygunluk skoru ve uyari sistemini de etkiliyor.

## Yapilacaklar

Asagidaki basliklar `proje_ozeti.docx` icindeki guncel yol haritasi baz alinarak derlenmistir.

### Modül 2 - Otomatik Kontrol, Skorlama ve Semantik Arama

- Rapor dil kalitesi, terminoloji tutarliligi ve eksik bulgu tespiti
- JCI akreditasyon kriterleriyle uyumlu otomatik skorlama
- Radyolog bazli performans metrikleri paneli
- Klinik denetim ve kalite guvencesi surecleri icin denetim logu
- Rapordan cikarilan bulgulara gore vakalari kritiklik skoruyla siralama
- Acil bildirimi gereken vakalari otomatik isaretleme
- Uyariyi sadece gostermek yerine is akisina yonlendirme
- FAISS veya pgvector tabanli benzer vaka arama
- "Bu bulgular hangi vakalarda daha once goruldu?" benzeri semantik arama
- Klinik arastirma ve egitim icin referans vaka bulma
- Radyolog gecmis raporlariyla kisisellestirilmis arama

### Modül 3 - Doktor Cumle Tamamlama Asistani

- Radyolog tek bir cumle soylediginde veya yazdiginda baglama uygun rapor metnini tamamlama
- BERTurk veya ince ayarli GPT-4o tabanli tamamlama modeli
- Hastane ve radyolog bazli kisisellestirme
- Hastanin gecmis tetkikleri, yas, cinsiyet ve sikayetiyle baglam zenginlestirme
- Onay/reddetme geri bildirimiyle surekli ogrenme dongusu
- Ilk karakterlerden sonra tamamlama onerileri sunan UX
- Tab tusu veya sesli onay ile kabul mekanizmasi
- Birden fazla tamamlama secenegi sunma

### Oncelikli Aksiyonlar

- Bir radyoloji profesörünü akademik danisman olarak projeye dahil etmek
- Modül 1'e odaklanmak; diger modulleri erken actirmamak
- Ozel hastane veri ortakligi icin dogrudan temas kurmak
- Mozilla Common Voice ve senaryo okuma ile ses verisi toplamak
- Pilot KPI'larini hastaneyle birlikte onceden tanimlamak
- Ilk odeme yapan musteri olmadan sonraki modullere gecmemek

## Repo Yapisi

```text
app/
  api/         FastAPI uygulamasi
  module1/     Rapor uretimi ve ASR akis kodlari
  module2/     Ollama tabanli degerlendirme ve uyari bilesenleri
  prompts/     LLM prompt dosyalari
data/
  raw/         Ham veri ve lokalde tutulan girdiler
  processed/   Islenmis veri ve benchmark girdileri
docs/          Proje ozeti, referans raporlar ve stratejik belgeler
experiments/   Deneysel scriptler
scripts/       Yardimci veri cekme/ceviri scriptleri
```

## Calistirma

Ortam aktifken:

```bash
uvicorn app.api.module1_api:app --reload
```

Tarayicida:

```text
http://127.0.0.1:8000/
```

## Not

Bu repo aktif gelistirme halindedir. Ana odak, sesli dikte ile gelen Turkce radyoloji rapor akisini daha dayanikli hale getirmek ve bunu gercek kullanim kosullarinda olcebilir bir urune donusturmektir.
