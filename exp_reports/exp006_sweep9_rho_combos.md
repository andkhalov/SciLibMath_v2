# EXP-006: Sweep 9 — rho=0.3, combos, contrast boost (20% data, 3 epochs)

> **Дата:** 2026-03-14
> **Ветка:** main
> **Предыдущий:** EXP-005 (sweep8, VA calibration)

## Мотивация

Sweep8 установил:
1. **E8c** (nonlinear MLP, alpha=0.01, init_scale=0.1) — best cm_R@1=0.3808
2. **VA calibration** помогает только static experiments, controller сам компенсирует
3. **Alignment saturation** — D_intra=0.007, alignment gradient≈0 при rho=0.1

**Нерешённая проблема:** rho=0.1 но D_intra=0.007 (14× ниже цели). Alignment насытился — увеличение λ_align не поможет, нужно увеличить radial target rho=0.3.

**Новый вопрос:** повышение приоритета contrast (contrast_weight=2.0) vs снижение VA — разные динамики, нужно сравнить.

## Дизайн эксперимента

### Новые эксперименты

| Конфиг | Базируется на | Изменение | Гипотеза |
|---|---|---|---|
| e6_rho03 | E6 | rho: 0.1→0.3 | D_intra вырастет, alignment оживёт |
| e8c_rho03 | E8c | rho: 0.1→0.3 | Best controller + alignment fix |
| e8c_low_va | E8c | lambda_va: 0.1→0.01 | Combo двух лучших находок |
| e8c_rho03_low_va | E8c | rho=0.3 + lambda_va=0.01 | Тройная комбинация |
| e8c_boost | E8c | contrast_weight: 2.0 | Приоритет contrast над VA |

### Контрольные

| Конфиг | Sweep8 baseline |
|---|---|
| e6_fuzzy | cm_R@1 = 0.3693 (sweep8) |
| e8c_active | cm_R@1 = 0.3808 (sweep8) |
| e10_potential_fuzzy | cm_R@1 = 0.3731 (sweep8) |

### Гипотезы

#### H26: rho=0.3 увеличит D_intra для E6
**Метрика:** D_intra
**Ожидание:** D_intra(E6_rho03) > 0.05 (vs E6=0.027)
**Уверенность:** Высокая

#### H27: e8c_rho03 будет best overall
**Метрика:** mean_crossmodal_R@1
**Ожидание:** E8c_rho03 > E8c (0.3808)
**Уверенность:** Средняя

#### H28: e8c_low_va ≈ E8c (controller compensates)
**Метрика:** mean_crossmodal_R@1
**Ожидание:** |E8c_low_va - E8c| < 0.5pp (controller already manages VA)
**Уверенность:** Средне-высокая

#### H29: contrast_weight=2.0 улучшит retrieval
**Метрика:** mean_crossmodal_R@1
**Ожидание:** E8c_boost > E8c. Больший gradient от contrast → лучший retrieval
**Уверенность:** Средне-низкая

#### H30: rho=0.3 НЕ ухудшит text→text retrieval
**Метрика:** en→ru R@1
**Ожидание:** en→ru R@1(E6_rho03) ≥ 0.83 (vs E6=0.842)
**Уверенность:** Средняя

## Изменения кода

| Файл | Строк | Backward-compatible |
|---|---|---|
| code/losses/composite.py | +3 (contrast_weight param) | ✅ default=1.0 |
| code/train.py | +6 (new names in dispatch) | ✅ |
| configs/e6_rho03.yaml | new | — |
| configs/e8c_rho03.yaml | new | — |
| configs/e8c_low_va.yaml | new | — |
| configs/e8c_rho03_low_va.yaml | new | — |
| configs/e8c_boost.yaml | new | — |
| run_ablation.sh | +7 (EXPERIMENTS_D) | ✅ |

## Sweep configuration

```
Data: 20%, batch_size=64, epochs=3
Experiments: 8 total (5 new + 3 controls)
Overrides: data.dataset_fraction=0.2 data.batch_size=64 training.epochs=3
Expected: ~6-7 hours
```

## Примечание: E10 и rho

rho не влияет на E10 (potential loss заменяет alignment+radial). Поэтому e10_rho03 не создан. E10 включён только как контроль.

## Результаты

> **Ожидание результатов:** ...

