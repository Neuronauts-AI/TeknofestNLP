# Deneysel Kalite Kontrol Modulu

Bu klasor, ana uygulamadan bagimsiz olarak rapor kalite kontrolu ve skorlama denemeleri icin ayrilmistir.

Amac:

- mevcut Turkce rapor metinlerini otomatik degerlendirmek
- alt skorlar uretmek
- eksik veya zayif alanlari kisa notlarla raporlamak
- sonuc tatmin ediciyse ana uygulamaya tasimak

Bu modulte su anda Ollama tabanli tek script vardir:

```bash
python experiments\quality_control\run_quality_control.py --findings "..." --impression "..."
```

Desteklenen ciktilar:

- `overall_label`
- `overall_score`
- `dil_kalitesi`
- `terminoloji_tutarliligi`
- `yapi_uygunlugu`
- `sonuc_yeterliligi`
- `issues`
- `summary`

Not:

- Bu klasor deneyseldir.
- Ana uygulama bu modulu henuz kullanmaz.
