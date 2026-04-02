# Deneysel Kalite Kontrol Modülü

Bu klasör, ana uygulamadan bağımsız olarak rapor kalite kontrolü ve skorlama denemeleri için ayrılmıştır.

Amaç:

- mevcut Türkçe rapor metinlerini otomatik değerlendirmek
- alt skorlar üretmek
- eksik veya zayıf alanları kısa notlarla raporlamak
- sonuç tatmin ediciyse ana uygulamaya taşımak

Bu modüldeki kalibre edilmiş sürüm artık ana uygulamada da kullanılmaktadır. Buradaki script,
toplu test ve ayrı denemeler için korunur.

```bash
python experiments\quality_control\run_quality_control.py --findings "..." --impression "..."
```

Desteklenen çıktılar:

- `overall_label`
- `overall_score`
- `dil_kalitesi`
- `terminoloji_tutarliligi`
- `yapi_uygunlugu`
- `sonuc_yeterliligi`
- `issues`
- `summary`
