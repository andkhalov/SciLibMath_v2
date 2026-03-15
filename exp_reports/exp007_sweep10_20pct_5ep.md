# EXP-007: Sweep 10 — Full ablation with live controller (20% data, 5 epochs)

> **Дата:** 2026-03-15
> **Ветка:** main
> **Предыдущий:** EXP-006 (sweep9, rho=0.3 + combos)
> **Commit:** 30c13c4

## Мотивация

Sweep10 — **первый sweep с живым fuzzy controller** после bugfix в EXP-006. Все предыдущие sweep'ы (7-9) работали с мёртвым контроллером (step_count freeze на step ~210).

**Цели:**
1. Полный ablation E1-E10 + Family B с рабочим контроллером
2. Верифицировать: живой контроллер > мёртвый?
3. Проверить E7 (Lyapunov) с живым контроллером
4. Подтвердить Family A > B gap

## Дизайн эксперимента

### Параметры
- **Данные:** 20% (37,764 train / 1,987 test)
- **Эпохи:** 5
- **Batch size:** 64
- **Seed:** 42
- **eval_every_steps:** 200

### 14 экспериментов

| Family | Эксперименты |
|--------|-------------|
| A (11) | E1, E2, E3, E4, E5, E6, E7, E8c, E9, E10, E8c_low_va |
| B (3)  | E1b, E3b, E4b |

## Результаты

### Сводная таблица (по mean_crossmodal_R@1)

| Rank | Exp | Family | cm_R@1 | cm_R@10 | D_intra | img→lat R@1 | Δ vs sweep9 |
|------|-----|--------|--------|---------|---------|-------------|-------------|
| 1 | E1 pairwise | A | **0.4705** | 0.6979 | 0.7367 | 0.3914 | — |
| 2 | E1b pairwise | B | 0.4465 | 0.6724 | 0.6289 | 0.3311 | — |
| 3 | E10 pot+fuzzy | A | 0.3972 | 0.5477 | 0.0143 | 0.0577 | +0.024 |
| 4 | E6 fuzzy | A | 0.3807 | 0.5291 | 0.0037 | 0.0386 | +0.011 |
| 5 | E7 lyapunov | A | 0.3803 | 0.5285 | 0.0037 | 0.0352 | — |
| 6 | E8c active | A | 0.3758 | 0.5250 | 0.0038 | 0.0359 | -0.005 |
| 7 | E8c low_va | A | 0.3758 | 0.5234 | 0.0040 | 0.0317 | -0.006 |
| 8 | E9 potential | A | 0.3710 | 0.5142 | 0.0114 | 0.0218 | — |
| 9 | E4 composite | A | 0.3556 | 0.5014 | 0.0036 | 0.0162 | — |
| 10 | E5 learnable | A | 0.3528 | 0.4982 | 0.0035 | 0.0139 | — |
| 11 | E3 centroid_reg | A | 0.3529 | 0.4972 | 0.0036 | 0.0136 | — |
| 12 | E4b composite | B | 0.3207 | 0.4608 | 0.0028 | 0.0070 | — |
| 13 | E3b centroid | B | 0.3204 | 0.4591 | 0.0027 | 0.0075 | — |
| 14 | E2 centroid | A | 0.0918 | 0.1948 | 1.5457 | 0.0006 | — |

### Ключевые наблюдения

1. **E1 pairwise = лучший** (cm_R@1=0.471). Pairwise по-прежнему доминирует.
2. **E10 potential+fuzzy = лучший centroid-based** (0.397). Potential loss с контроллером обходит E6.
3. **E6 fuzzy с живым контроллером** (0.381) — хуже мёртвого E6 (sweep7: 0.419). Drift ±0.001-0.008 слишком мал.
4. **E7 ≈ E6** (0.380 vs 0.381) — Lyapunov constraint не даёт пользы.
5. **E2 по-прежнему катастрофа** (0.092) — центроидный InfoNCE без регуляризации не работает.
6. **Family A > B** на +2.4pp (E1: 0.471 vs E1b: 0.447), +3.3pp (E3: 0.353 vs E3b: 0.320), +3.5pp (E4: 0.356 vs E4b: 0.321).

### Контроллер: живой, но слабый

1444 controller events за весь sweep. Траектория нестационарна, но drift ±0.001-0.008 — практически неподвижен. **Причина:** α/γ = 0.001/0.01 = 0.1 — elastic reversion (γ) в 10× сильнее шага (α). Контроллер не может отойти от defaults.

### Loss component analysis

L_va по-прежнему доминирует: ~75-79% total loss budget для centroid-based экспериментов. Contrast + alignment + radial = <25%.

## Проверка гипотез H31-H40

| # | Гипотеза | Вердикт | Доказательство |
|---|----------|---------|----------------|
| H31 | Живой E6 > мёртвый E6 (sweep7: 0.419) | **ОТВЕРГНУТА** | E6=0.381 < 0.419. Живой контроллер с α=0.001 ХУЖЕ |
| H32 | E8c (nonlinear) > E6 (linear) | **ОТВЕРГНУТА** | E8c=0.376 < E6=0.381 |
| H33 | E8c_low_va = best overall | **ОТВЕРГНУТА** | E8c_low_va=0.376, не лучший. E1=0.471 best |
| H34 | E7 > E6 при живом контроллере | **ОТВЕРГНУТА** | E7=0.380 ≈ E6=0.381 (нет пользы) |
| H35 | E9 > E3 по D_intra | **ПОДТВЕРЖДЕНА** | D_intra: E9=0.011 vs E3=0.004 (+175%) |
| H36 | E10 > E9 | **ПОДТВЕРЖДЕНА** | E10=0.397 > E9=0.371 (+2.6pp) |
| H37 | Family A > B на 2-4pp | **ПОДТВЕРЖДЕНА** | E1: +2.4pp, E3: +3.3pp, E4: +3.5pp |
| H38 | Lambda trajectory нестационарна | **ЧАСТИЧНО** | Нестационарна, но drift ±0.001-0.008 |
| H39 | Rule activations > 0 после warmup | **ПОДТВЕРЖДЕНА** | 1444 controller events за sweep |
| H40 | E5 < E6 | **ПОДТВЕРЖДЕНА** | E5=0.353 < E6=0.381 (+2.8pp) |

**Итог:** 5 подтверждено, 4 отвергнуто, 1 частично.

## Выводы

1. **Controller работает, но слишком слаб.** α/γ баланс = 0.1 — elastic доминирует. Нужно α×5, γ÷5 → α/γ=2.5.
2. **L_va доминирует.** Визуальный alignment забирает 75-79% gradient budget. Нужен curriculum scheduling.
3. **E10 (potential+fuzzy) — лучший centroid-based.** Potential loss > alignment+radial для центроидной геометрии.
4. **E1 pairwise unchallenged.** Gap 7.3pp vs E10. Но pairwise не масштабируется на M модальностей — ценность centroid-based подхода в обобщаемости.
5. **LR schedule plateau.** weight_norm стагнирует к концу — final_div_factor=1e4 (default) слишком агрессивен.

## Следующие шаги (EXP-008)

1. **Family C:** OCR-pretrained visual encoder (TrOCR-small-printed DeiT-small) вместо ResNet18
2. **L_va curriculum:** warmup + linear ramp для визуального alignment
3. **Controller tuning:** α=0.005, γ=0.002 (α/γ=2.5)
4. **LR fix:** final_div_factor=10

---

*EXP-007 | SciLibMath_v2 | 2026-03-15*
