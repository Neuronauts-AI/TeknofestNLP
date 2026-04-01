# Ollama Report Style Check

Bu klasör ana uygulamadan bağımsızdır. Sadece deneme amaçlıdır ve sonradan kolayca silinebilir.

Amaç:
- Türkçe radyoloji rapor metnini
- örnek hastane/radyoloji raporu üslubuna göre
- `uygun` veya `uygun_degil` olarak sınıflandırmak

Model:
- `lfm2-8B-A1B:latest`

Referans:
- [docs/reference_report_style.txt](../../docs/reference_report_style.txt)
- Sınıflandırma bu dosyadaki örnek rapor üslubuna göre yapılır.

Çalıştırma:
```cmd
python experiments\ollama_style_check\classify_report_style_ollama.py --findings "Akciğer alanlarında belirgin konsolidasyon izlenmemiştir." --impression "Akut kardiyopulmoner patoloji saptanmamıştır."
```

Tek metin ile:
```cmd
python experiments\ollama_style_check\classify_report_style_ollama.py --text "Bulgular: Kardiyotorasik oran normal sınırlardadır. Sonuç: Normal sınırlarda akciğer grafisi incelemesi."
```

JSON çıktı verir:
- `label`
- `score`
- `reason`
- `raw_response`
