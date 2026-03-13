# EXP-005: Sweep 8 — VA calibration + E8 controller fix (20% data, 3 epochs)

> **Дата:** 2026-03-13
> **Ветка:** main
> **Предыдущий:** EXP-004 (sweep7, 14 экспериментов)

## Мотивация

Анализ абсолютных значений компонентов лосса в sweep7 показал:

| Компонент | E3 | E6 | E10 |
|---|---|---|---|
| λ_va × L_va | **76.9%** | **43.7%** | **37.9%** |
| Contrast | 21.2% | 50.7% | 48.6% |
| Alignment | 1.8% | 1.3% | 2.3% |

**Проблема 1:** L_va (visual alignment) доминирует в лоссе. Модель на E3-E5 на 77% оптимизирует выравнивание ResNet фичей, а не cross-modal retrieval.

**Проблема 2:** E8 controller мёртв — все lambda ≈ defaults (alpha=0.001 + init_scale=0.01 слишком консервативны).

## Дизайн эксперимента

### Принцип: freeze & extend
- Существующие E1-E10 конфиги НЕ изменены
- Код backward-compatible (новые params с defaults = текущее поведение)
- Новые конфиги с суффиксом `c` (calibration)

### Новые эксперименты

| Конфиг | Базируется на | Изменение | Гипотеза |
|---|---|---|---|
| e3c_low_va | E3 | lambda_va: 0.1→0.01 | VA 77%→~20%, contrast/align получат больше gradient'а |
| e6c_low_va | E6 | lambda_va: 0.1→0.01 | VA 44%→~8%, controller адаптируется к новому балансу |
| e10c_low_va | E10 | lambda_va: 0.1→0.01 | Best centroid + reduced VA dominance |
| e8c_active | E8 | alpha: 0.001→0.01, init_scale: 0.01→0.1 | Controller оживёт (10× step, 10× init) |

### Контрольные (для прямого сравнения)

| Конфиг | Sweep7 baseline |
|---|---|
| e3_centroid_reg | cm_R@1 = 0.3523 |
| e6_fuzzy | cm_R@1 = 0.4188 |
| e10_potential_fuzzy | cm_R@1 = 0.4222 |
| e8_nonlinear | cm_R@1 = 0.3580 |

### Гипотезы

#### H21: lambda_va=0.01 улучшит E3 crossmodal
**Метрика:** mean_crossmodal_R@1
**Ожидание:** E3c > E3 (0.3523). При меньшей VA dominance, contrast/align получат более сильный gradient signal
**Уверенность:** Средняя

#### H22: lambda_va=0.01 улучшит E6 crossmodal
**Метрика:** mean_crossmodal_R@1
**Ожидание:** E6c > E6 (0.4188). Controller уже поднимает τ, с меньшим VA давлением может быть ещё эффективнее
**Уверенность:** Средняя

#### H23: E10c будет best overall centroid-based
**Метрика:** mean_crossmodal_R@1
**Ожидание:** E10c > E10 (0.4222)
**Уверенность:** Средняя

#### H24: E8c controller будет активен
**Метрика:** |τ_final - τ_default| > 0.01, std(lambda_t) > E8
**Ожидание:** E8c lambdas сдвинутся от defaults (в отличие от E8 в sweep7)
**Уверенность:** Высокая

#### H25: E8c crossmodal ≥ E6
**Метрика:** mean_crossmodal_R@1
**Ожидание:** E8c ≥ 0.4188 (E6 level)
**Уверенность:** Средне-низкая

## Sweep configuration

```
Data: 20%, batch_size=64, epochs=3
Experiments: 8 total (4 new + 4 controls)
Overrides: data.dataset_fraction=0.2 data.batch_size=64 training.epochs=3
Expected: ~10 hours
```

## Изменения кода

| Файл | Строк | Backward-compatible |
|---|---|---|
| code/controller/rules.py | +2 (init_scale param) | ✅ default=0.01 |
| code/controller/ts_controller.py | +2 (pass init_scale) | ✅ |
| code/train.py | +5 (c-variants in dispatch) | ✅ |
| configs/e3c_low_va.yaml | new | — |
| configs/e6c_low_va.yaml | new | — |
| configs/e10c_low_va.yaml | new | — |
| configs/e8c_active.yaml | new | — |
| run_ablation.sh | +7 (EXPERIMENTS_C) | ✅ |

## Результаты

### Сводная таблица

| Exp | mean_cm_R@1 | cent_R@1 | img→cent | D_intra | VA% | Notes |
|-----|-------------|----------|----------|---------|-----|-------|
| e3 (ctrl) | | | | | | sweep7 baseline |
| e3c | | | | | | lambda_va=0.01 |
| e6 (ctrl) | | | | | | sweep7 baseline |
| e6c | | | | | | lambda_va=0.01 |
| e10 (ctrl) | | | | | | sweep7 baseline |
| e10c | | | | | | lambda_va=0.01 |
| e8 (ctrl) | | | | | | sweep7 baseline |
| e8c | | | | | | alpha=0.01, init=0.1 |

### H21-H25 verification
<!-- Fill after sweep completes -->

## Проблемы и наблюдения
<!-- Fill after sweep -->

## Следующие шаги
<!-- Fill after analysis -->
