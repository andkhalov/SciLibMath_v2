# EXP-004: Sweep 7 — E8-E10 + corrections (20% data, 3 epochs)

> **Дата:** 2026-03-12
> **Ветка:** model/SciLibMath_v2
> **Предыдущий:** EXP-003 (sweep6, 20%/5ep)

## Изменения относительно EXP-003

### Новые эксперименты
- **E8 (nonlinear T-S):** MLP consequents φ_r(s_t) вместо линейных A_r·s_t + b_r
- **E9 (potential + learnable W):** U_attract + U_repel вместо L_align + L_rad, LossMixer
- **E10 (potential + fuzzy):** U_attract + U_repel + fuzzy controller

### Корректировки параметров
- **rho:** 0.1 → 0.3 (base.yaml) — ослабить radial over-compression
- **Lyapunov xi:** 0.01 → 0.001 — жёстче штрафовать рост V_t
- **Lyapunov penalty_weight:** 0.1 → 0.5 — сильнее влияние Lyapunov

## Гипотезы

### H15: rho 0.1→0.3 увеличит D_intra
**Метрика:** D_intra для E3-E5
**Ожидание:** 0.007 → ~0.1-0.2
**Уверенность:** Высокая

### H16: E8 nonlinear consequents расширят диапазон λ_t
**Метрика:** std(λ_t) за эпоху, min/max по компонентам
**Ожидание:** λ_t двигается дальше от defaults чем E6
**Уверенность:** Средне-высокая

### H17: E9 potential loss ≥ E3 по img→centroid R@1
**Метрика:** img→centroid R@1
**Ожидание:** ≥ 0.94 (E3 level)
**Уверенность:** Средняя

### H18: E9 лучше E5 по D_inter
**Метрика:** D_inter (межцентроидное расстояние)
**Ожидание:** > E5 (log-barrier repulsion → лучшее разделение)
**Уверенность:** Средне-высокая

### H19: Lyapunov tighter (xi=0.001, pw=0.5) создаст gap E7>E6
**Метрика:** variance of training curves, mean_crossmodal_R@1
**Ожидание:** E7 > E6 (было E7 ≈ E6)
**Уверенность:** Средне-высокая

### H20: E10 = best among E8-E10 (exploratory)
**Метрика:** mean_crossmodal_R@1
**Ожидание:** E10 > E9 > E8 (speculative)
**Уверенность:** Низкая

## Sweep configuration

```
Data: 20%, batch_size=64, epochs=3
Experiments: 14 total (E1-E10 Family A + E1b-E4b Family B)
Overrides: data.dataset_fraction=0.2 data.batch_size=64 training.epochs=3
```

## Результаты

### Сводная таблица

| Exp | mean_cm_R@1 | cent_R@1 | img→cent_R@1 | D_intra | D_inter | collapse | Notes |
|-----|-------------|----------|--------------|---------|---------|----------|-------|
| E1  |             |          |              |         |         |          |       |
| E2  |             |          |              |         |         |          |       |
| E3  |             |          |              |         |         |          |       |
| E4  |             |          |              |         |         |          |       |
| E5  |             |          |              |         |         |          |       |
| E6  |             |          |              |         |         |          |       |
| E7  |             |          |              |         |         |          |       |
| E8  |             |          |              |         |         |          | NEW   |
| E9  |             |          |              |         |         |          | NEW   |
| E10 |             |          |              |         |         |          | NEW   |
| E1b |             |          |              |         |         |          |       |
| E2b |             |          |              |         |         |          |       |
| E3b |             |          |              |         |         |          |       |
| E4b |             |          |              |         |         |          |       |

### H15-H20 verification
<!-- Fill after sweep completes -->

## Проблемы и наблюдения
<!-- Fill after sweep -->

## Следующие шаги
<!-- Fill after analysis -->
