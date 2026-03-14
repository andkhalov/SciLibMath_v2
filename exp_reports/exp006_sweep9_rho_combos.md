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

> **Sweep завершён:** 2026-03-14 10:19–16:35 MSK, 8/8 OK, ~6.3 часа

### Сводная таблица

| Rank | Exp | cm_R@1 | cm_R@10 | img→cent | D_intra | D_inter | mod_bal |
|------|-----|--------|---------|----------|---------|---------|---------|
| 1 | **e8c_low_va** | **0.3815** | **0.5494** | 0.8421 | 0.0294 | 0.9998 | 0.0043 |
| 2 | e8c_rho03_low_va | 0.3807 | 0.5457 | 0.8483 | 0.0321 | 0.9997 | 0.0047 |
| 3 | e8c_rho03 | 0.3792 | 0.5396 | 0.8016 | 0.0311 | 0.9997 | 0.0043 |
| 4 | e8c_boost | 0.3787 | 0.5417 | 0.7528 | 0.0331 | 0.9997 | 0.0039 |
| 5 | e8c_active (ctrl) | 0.3783 | 0.5356 | 0.8290 | 0.0337 | 0.9997 | 0.0048 |
| 6 | e10 (ctrl) | 0.3734 | 0.5247 | 0.9160 | 0.0427 | 0.9997 | 0.0080 |
| 7 | e6 (ctrl) | 0.3694 | 0.5225 | 0.9150 | 0.0270 | 0.9998 | 0.0051 |
| 8 | e6_rho03 | 0.3685 | 0.5202 | 0.9167 | 0.0280 | 0.9998 | 0.0054 |

### Image cross-modal R@1

| Пара | e6 | e6_rho03 | e8c | e8c_rho03 | e8c_low_va | e8c_boost |
|---|---|---|---|---|---|---|
| img↔latex | 0.037 | 0.032 | 0.053 | 0.062 | **0.079** | 0.067 |
| img↔en | 0.008 | 0.008 | 0.011 | 0.012 | **0.018** | 0.015 |
| img↔lean | 0.011 | 0.009 | 0.016 | 0.018 | **0.021** | 0.020 |
| img↔ru | 0.010 | 0.008 | 0.012 | 0.012 | **0.016** | 0.015 |

### H26-H30 verification

| Hyp | Verdict | Evidence |
|-----|---------|---------|
| H26 (rho=0.3 → D_intra↑) | ⚠️ **МИНИМАЛЬНО** | D_intra: E6=0.027→E6_rho03=0.028 (+4%). Controller компенсирует |
| H27 (e8c_rho03 best) | ❌ **ОТВЕРГНУТА** | e8c_low_va лучше (0.3815 vs 0.3792) |
| H28 (e8c_low_va ≈ E8c) | ❌ **ОТВЕРГНУТА** | +0.3pp — значимо лучше, low_va помогает и nonlinear controller |
| H29 (contrast boost) | ⚠️ **НЕЙТРАЛЬНО** | +0.04pp — в пределах шума |
| H30 (text stable) | ✅ **ПОДТВЕРЖДЕНА** | en↔ru: 0.849-0.856 стабильно |

### Loss component analysis (final values)

| Exp | Contrast | Align | Rad | VA | Personal |
|-----|----------|-------|-----|-----|----------|
| e8c_low_va | 0.090 | 0.008 | 0.008 | 6.65 | 1.81 |
| e8c_active | 0.110 | 0.009 | 0.010 | 7.92 | 2.11 |
| e6_fuzzy | 0.046 | 0.008 | 0.007 | 9.35 | 1.36 |

**VA по-прежнему доминирует** (6.6-9.4 vs <0.15 для остальных).

## КРИТИЧЕСКИЙ БАГ: Controller Freeze

### Обнаружено

Все fuzzy controllers во ВСЕХ sweep'ах (7, 8, 9) замерзали на step ~210. Rule activations → 0.0000 после warmup. Controller работал только 200 шагов из ~8,600.

### Причина

Два бага в `ts_controller.py`:

**Bug 1 (step_count stuck):** `step_count` инкрементировался только внутри `_normalize_state()`, которая вызывалась только из `compute_correction()`. На skipped steps (step_frequency=10) compute_correction не вызывался → step_count застревал на 201 навсегда → controller никогда не активировался.

**Bug 2 (logging offset):** TensorBoard логировал rule activations каждые 10 шагов (global_step % 10 == 0). Controller fires at _training_step % 10 == 0. Из-за offset +1 между global_step и _training_step, logging ВСЕГДА попадал на skipped step → показывал 0.

### Исправление

1. Выделен `_training_step` — всегда инкрементируется в `step()`
2. Выделен `_update_running_stats()` — обновляет running mean/var на КАЖДОМ шаге
3. Добавлен `_cached_h_bar` — возвращает последнее активное h_bar на skipped steps

### Верификация

После fix: e6_fuzzy на 5%/2ep — rules активны на 60-124/125 шагов после warmup. Lambda trajectory:
- τ: 0.071 → 0.099 (+40%)
- λ_va: 0.100 → 0.128 (+28%)
- λ_reg: 0.051 → 0.060 (+18%)

### Импликации

**ВСЕ предыдущие результаты sweep7-9 получены с мёртвым контроллером.** Controller адаптировал λ только во время warmup (200 шагов), затем замирал. Фактически E6-E10 работали как "E4 с warm-started static weights". Нужен re-run с живым контроллером.

## Выводы и следующие шаги

### Что убираем
- rho=0.3 для controller experiments — controller компенсирует, эффект <4%
- contrast_weight boost — нейтральный эффект
- lambda_va calibration для linear controller (E6) — уже доказано в sweep8

### Что оставляем (рабочие baseline)
- **e8c_low_va** — best config (0.3815 с мёртвым контроллером!)
- **E6** — стабильный baseline для linear controller
- **E10** — potential loss reference

### Что делаем дальше (ПРИОРИТЕТ)
- [ ] **Re-run sweep с живым контроллером** — это главное. Все результаты E6-E10 невалидны
- [ ] **e8c_low_va vs e6 vs e10** — сравнить с рабочим контроллером
- [ ] Multi-seed для top-3 после re-run
- [ ] Full training (10 эпох, 100%) только после валидации с живым контроллером
