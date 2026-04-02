# Deneysel Semantik Arama

Bu modül, Türkçeye çevrilmiş radyoloji raporlarında benzer vaka araması yapmak için
Ollama üzerindeki `qwen3-embedding:0.6b` modelini kullanır.

Şimdilik ana uygulamaya bağlı değildir. Önce deneysel olarak test edilir.

## Ne yapar?

- CSV içindeki raporları embed eder
- Embedding indeksini JSON dosyasına yazar
- Sorgu metni için en benzer raporları döndürür

## Varsayılan veri kaynağı

- `data/processed/mimic_cxr_text_only_tr_gemini.csv`

Beklenen kolonlar:

- `findings_tr` / `impression_tr`
- veya fallback olarak `findings` / `impression`

## İndeks oluşturma

```cmd
python experiments\semantic_search\run_semantic_search.py build --csv data\processed\mimic_cxr_text_only_tr_gemini.csv --output data\processed\semantic_search_index.json
```

## Sorgu çalıştırma

```cmd
python experiments\semantic_search\run_semantic_search.py query --index data\processed\semantic_search_index.json --text "pnömotoraks ve plevral efüzyon" --top-k 5
```

## Not

Bu modül gerçek klinik karar sistemi değildir. Amaç, benzer vaka bulma yaklaşımını
denemek ve çevrilmiş metinlerde semantik yakınlığın yeterince anlamlı olup olmadığını
görmektir.
