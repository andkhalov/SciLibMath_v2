# EXP-004: Sweep 7 — E8-E10 + corrections (20% data, 3 epochs)

> **Дата запуска:** 2026-03-12 13:54 MSK
> **Дата завершения:** 2026-03-13 07:33 MSK (~17.5 часов)
> **Ветка:** model/SciLibMath_v2
> **Коммит:** dca104d
> **Предыдущий:** EXP-003 (sweep6, 20%/5ep)
> **Статус:** 14/14 OK

## Изменения относительно EXP-003

### Новые эксперименты
- **E8 (nonlinear T-S):** MLP consequents φ_r(s_t) вместо линейных A_r·s_t + b_r (MATH.md M.3.8)
- **E9 (potential + learnable W):** U_attract + U_repel вместо L_align + L_rad, LossMixer (MATH.md M.3.9)
- **E10 (potential + fuzzy):** U_attract + U_repel + fuzzy controller (MATH.md M.3.10)

### Корректировки параметров
- ~~**rho:** 0.1 → 0.3 (base.yaml)~~ **⚠️ НЕ ПРИМЕНЕНО** — пропущено при реализации, все эксперименты запущены с rho=0.1
- **Lyapunov xi:** 0.01 → 0.001 ✅
- **Lyapunov penalty_weight:** 0.1 → 0.5 ✅

## Гипотезы

### H15: rho 0.1→0.3 увеличит D_intra
**Метрика:** D_intra для E3-E5
**Ожидание:** 0.007 → ~0.1-0.2
**Результат:** ⚠️ **НЕ ТЕСТИРОВАЛОСЬ** — rho осталось 0.1, D_intra = 0.0074 (E3), 0.0078 (E4), 0.0073 (E5) — без изменений.

### H16: E8 nonlinear consequents расширят диапазон λ_t
**Метрика:** std(λ_t), crossmodal vs E6
**Ожидание:** λ_t двигается дальше от defaults, crossmodal > E6
**Результат:** ❌ **ОТВЕРГНУТА.** E8 mean_cm_R@1 = 0.3580 vs E6 = 0.4188. E8 ведёт себя как E4 (static composite), D_intra = 0.0078 — идентично E4. Controller в E8 не адаптируется.

### H17: E9 potential loss ≥ E3 по img→centroid R@1
**Метрика:** img→centroid R@1
**Ожидание:** ≥ 0.94 (E3 level)
**Результат:** ✅ **ПОДТВЕРЖДЕНА.** E9 img→centroid R@1 = 0.9427 ≈ E3 (0.9423). Potential loss не хуже radial для centroid alignment.

### H18: E9 лучше E5 по D_inter
**Метрика:** D_inter (межцентроидное расстояние)
**Ожидание:** > E5 (log-barrier repulsion → лучшее разделение)
**Результат:** ⚠️ **НЕЙТРАЛЬНО.** E9 D_inter = 0.9995, E5 = 0.9989. Разница ~0.0006, оба near 1.0. Log-barrier repulsion не дал заметного эффекта на D_inter (уже на гиперсфере = максимум).

### H19: Lyapunov tighter (xi=0.001, pw=0.5) создаст gap E7>E6
**Метрика:** mean_crossmodal_R@1
**Ожидание:** E7 > E6
**Результат:** ❌ **ОТВЕРГНУТА.** E7 = 0.4137, E6 = 0.4188. Ужесточение Lyapunov УХУДШИЛО результат на 0.51%. Lyapunov constraint слишком сильно ограничивает controller (penalty_weight=0.5 подавляет адаптацию).

### H20: E10 = best among E8-E10 (exploratory)
**Метрика:** mean_crossmodal_R@1
**Ожидание:** E10 > E9 > E8
**Результат:** ✅ **ПОДТВЕРЖДЕНА.** E10 = 0.4222 > E9 = 0.3704 > E8 = 0.3580. E10 (potential + fuzzy) — лучший среди новых, и лучший среди ВСЕХ centroid-based (> E6 = 0.4188).

## Sweep configuration

```
Data: 20%, batch_size=64, epochs=3
Experiments: 14 total (E1-E10 Family A + E1b-E4b Family B)
Overrides: data.dataset_fraction=0.2 data.batch_size=64 training.epochs=3
Duration: ~17.5 hours, ~1.25h per experiment
```

## Результаты

### Сводная таблица

| Exp | mean_cm_R@1 | cent_R@1 | img→cent | D_intra | D_inter | collapse | mod_bal | Notes |
|-----|-------------|----------|----------|---------|---------|----------|---------|-------|
| E1  | **0.4695**  | 0.8641   | 0.8312   | 1.4076  | 0.9829  | 0.0171   | 0.5566  | best cm, worst geometry |
| E2  | 0.0928      | 0.9601   | 0.9292   | 2.1369  | 0.9998  | 0.0002   | 0.0822  | centroid collapse |
| E3  | 0.3523      | 0.9601   | 0.9423   | 0.0074  | 0.9997  | 0.0003   | 0.0012  | baseline centroid+reg |
| E4  | 0.3550      | 0.9606   | 0.9418   | 0.0078  | 0.9998  | 0.0002   | 0.0012  | ≈ E3 |
| E5  | 0.3533      | 0.9602   | 0.9426   | 0.0073  | 0.9989  | 0.0011   | 0.0011  | ≈ E3 |
| E6  | 0.4188      | 0.9211   | 0.9013   | 0.0163  | 0.9993  | 0.0007   | 0.0032  | fuzzy lifts cm |
| E7  | 0.4137      | 0.9230   | 0.9118   | 0.0168  | 0.9995  | 0.0005   | 0.0035  | lyap hurts slightly |
| E8  | 0.3580      | 0.9606   | 0.9425   | 0.0078  | 0.9998  | 0.0002   | 0.0012  | **NEW** ≈ E4, controller dead |
| E9  | 0.3704      | 0.9581   | 0.9427   | 0.0231  | 0.9995  | 0.0005   | 0.0035  | **NEW** pot > E3-E5 |
| E10 | **0.4222**  | 0.9193   | 0.8930   | 0.0349  | 0.9993  | 0.0007   | 0.0068  | **NEW** best centroid-based |
| E1b | 0.4447      | 0.8527   | 0.7729   | 1.2060  | 0.9705  | 0.0295   | 0.3553  | Fam B < Fam A |
| E2b | 0.1850      | 0.9373   | 0.9294   | 1.7548  | 0.9996  | 0.0004   | 0.1687  | |
| E3b | 0.3213      | 0.9519   | 0.9410   | 0.0061  | 0.9995  | 0.0005   | 0.0011  | |
| E4b | 0.3213      | 0.9526   | 0.9429   | 0.0066  | 0.9993  | 0.0007   | 0.0011  | |

### Ранжирование по mean_crossmodal_R@1

```
1. E1  (pairwise)         0.4695  ← overall best, но без centroid geometry
2. E1b (pairwise FamB)    0.4447
3. E10 (potential+fuzzy)   0.4222  ← BEST centroid-based ★
4. E6  (fuzzy)            0.4188
5. E7  (lyapunov)         0.4137
6. E9  (potential)         0.3704  ← best among E3-E5-E9 (без controller)
7. E8  (nonlinear)        0.3580  ← controller не работает
8. E4  (composite static) 0.3550
9. E5  (composite learn)  0.3533
10. E3  (centroid+reg)     0.3523
11. E3b (centroid+reg B)   0.3213
12. E4b (composite B)      0.3213
13. E2b (centroid B)       0.1850
14. E2  (centroid)         0.0928
```

### Ключевые crossmodal пары (R@1)

| Pair | E1 | E3 | E6 | E8 | E9 | E10 |
|------|-----|-----|-----|-----|-----|------|
| en→ru | 0.856 | 0.850 | 0.868 | 0.853 | 0.864 | 0.873 |
| ru→en | 0.878 | 0.876 | 0.886 | 0.878 | 0.889 | 0.888 |
| en→lean | 0.728 | 0.659 | 0.737 | 0.664 | 0.691 | 0.747 |
| lean→en | 0.718 | 0.658 | 0.737 | 0.666 | 0.685 | 0.745 |
| latex→lean | 0.650 | 0.543 | 0.637 | 0.562 | 0.578 | 0.645 |
| **img→latex** | **0.297** | **0.012** | **0.108** | **0.018** | **0.024** | **0.103** |
| **img→en** | **0.085** | **0.003** | **0.021** | **0.002** | **0.005** | **0.018** |

### D_intra по модальностям

| Modality | E1 | E3 | E6 | E8 | E9 | E10 |
|----------|------|--------|--------|--------|--------|--------|
| en | 0.744 | 0.004 | 0.008 | 0.004 | 0.012 | 0.016 |
| ru | 0.726 | 0.004 | 0.008 | 0.004 | 0.012 | 0.017 |
| lean | 0.723 | 0.005 | 0.012 | 0.006 | 0.015 | 0.025 |
| latex | 0.661 | 0.006 | 0.010 | 0.007 | 0.018 | 0.022 |
| **img** | **4.185** | **0.019** | **0.044** | **0.019** | **0.059** | **0.095** |

## H15-H20 Verification Summary

| Hyp | Verdict | Key evidence |
|-----|---------|-------------|
| H15 | ⚠️ NOT TESTED | rho change missed, D_intra unchanged |
| H16 | ❌ REJECTED | E8 ≈ E4 (0.358 vs 0.355), controller dead |
| H17 | ✅ CONFIRMED | E9 img→cent = 0.943 ≈ E3 (0.942) |
| H18 | ⚠️ NEUTRAL | D_inter nearly identical (0.999x for all) |
| H19 | ❌ REJECTED | E7 < E6 (0.414 vs 0.419), lyap hurts |
| H20 | ✅ CONFIRMED | E10 > E9 > E8 (0.422 > 0.370 > 0.358) |

## Ключевые наблюдения

### 1. Fuzzy controller — единственный механизм, поднимающий crossmodal

Чёткий паттерн: E3/E4/E5 дают ~0.35 mean_cm_R@1, а E6/E10 дают ~0.42. Разница +20% приходит ТОЛЬКО от fuzzy controller. LossMixer (E5) и nonlinear consequents (E8) не помогают.

### 2. E8 nonlinear controller НЕ работает

E8 идентичен E4 по всем метрикам (D_intra=0.0078 vs 0.0078, cm_R@1=0.358 vs 0.355). MLP consequents инициализированы near-zero и не обучаются достаточно за 3 эпохи. Возможные причины:
- Init too conservative (`weight.mul_(0.01)`)
- alpha=0.001 clamp → nonlinear output зажат в том же range что линейный
- 3 эпохи недостаточно для обучения MLP params через indirect gradients

### 3. Potential loss (E9) > composite (E3-E5), но без controller

E9 mean_cm_R@1 = 0.3704 — лучше E3-E5 (~0.353). D_intra_img = 0.059 (vs 0.019 у E3), D_intra в 3× выше. Potential loss не давит embeddings в точку, а сохраняет более здоровую геометрию.

### 4. E10 = new SOTA для centroid-based

E10 (potential + fuzzy) = 0.4222 — лучший centroid-based результат, обходит E6 (0.4188). D_intra = 0.035 (vs E6: 0.016) — более рыхлая структура при лучшем crossmodal. Potential loss + fuzzy controller = выигрышная комбинация.

### 5. Lyapunov вредит (E7 < E6)

С ужесточёнными параметрами (xi=0.001, pw=0.5) разрыв E7-E6 стал -0.51% (было ~0%). Lyapunov constraint активно подавляет адаптацию controller: penalty_weight=0.5 слишком много для 3 эпох.

### 6. Image modality — узкое место

D_intra_img всегда в 3-5× больше остальных. img→text R@1 остаётся <0.03 для non-controller, <0.11 для controller экспериментов. Fuzzy controller существенно помогает (img→latex: 0.01→0.10), но image всё равно отстаёт.

### 7. Family B < Family A систематически

E1b < E1 на 2.5%, E3b < E3 на 3.1%, E4b < E4 на 3.4%. Два энкодера (symbolic + visual) стабильно хуже пяти отдельных.

## Проблемы

1. **rho=0.3 НЕ ПРИМЕНЁН** — нужно исправить и перезапустить минимум E3-E7 для верификации H15
2. **E8 controller dead** — нужна диагностика: larger init, higher alpha, или longer training
3. **D_inter ≈ 1.0 для всех** — метрика неинформативна при нормализации на гиперсфере. Нужна более чувствительная метрика разделения (minimum D_inter, или percentile)
4. **Lyapunov penalty_weight=0.5 слишком сильный** — попробовать 0.2 или 0.3

## Post-sweep анализ: декомпозиция лосса

### Абсолютные вклады компонентов (взвешенные, % от total)

| Компонент | E3 (0.573) | E6 (1.887) | E9 (1.028) | E10 (2.007) |
|---|---|---|---|---|
| **λ_va × L_va** | **0.441 (76.9%)** | **0.825 (43.7%)** | **0.645 (62.8%)** | **0.760 (37.9%)** |
| **Contrast (all)** | 0.122 (21.2%) | **0.956 (50.7%)** | 0.154 (15.0%) | **0.976 (48.6%)** |
| **Alignment (all)** | 0.010 (1.8%) | 0.024 (1.3%) | 0.030 (2.9%) | 0.047 (2.3%) |
| **Potential** | — | — | 0.199 (19.3%) | 0.142 (7.1%) |
| Global (rest) | 0.002 (0.4%) | 0.040 (2.1%) | 0.001 (0.1%) | 0.004 (0.2%) |

### Выводы

1. **L_va доминирует** (44-77% total) — модель на E3-E5 на 77% оптимизирует visual alignment, не cross-modal retrieval
2. **Alignment насытился** — D_intra=0.007, sq_dist→0, gradient→0. Alignment "победил" и остановился
3. **E6/E10 лучше** потому что controller поднимает τ (0.07→0.11), усиливая contrast с 21%→51% от total
4. **img_align в 12-14× больше text_align** — image всё ещё далеко от centroid
5. **Rule activations = 0 в логах** — артефакт step_frequency=10, controller реально работает (τ, weights сдвинуты)
6. **rho=0.3 не применён** — D_intra зажат у 0.007 вместо ожидаемых 0.1-0.2

### Controller lambda drift (defaults → final)

| Param | Default | E6 final | E8 final | E10 final |
|---|---|---|---|---|
| τ | 0.070 | **0.113** | 0.071 | **0.113** |
| λ_align | 0.300 | 0.303 | 0.300 | 0.397 |
| w_lean | 1.500 | 1.418 | 1.501 | 1.416 |
| w_latex | 1.500 | 1.423 | 1.500 | 1.421 |
| w_en | 1.000 | 0.927 | 0.999 | 0.925 |

E8 controller мёртв (все params ≈ default). E6/E10 — τ сдвинут +62%, lean/latex weights снижены -5%.

## Следующие шаги (EXP-005 план)

### Гипотеза 1: L_va доминирование вредит
**Действие:** Sweep с lambda_va ∈ {0.01, 0.03, 0.1 (current)} для E3 и E6
**Ожидание:** При lambda_va=0.01 contrast и alignment получат больше gradient'а → лучше crossmodal

### Гипотеза 2: rho=0.3 ослабит radial compression
**Действие:** Наконец применить rho=0.3, перезапустить E3-E7
**Ожидание:** D_intra ↑ от 0.007 к ~0.05-0.1, alignment gradient ≠ 0, больше модальной структуры

### Гипотеза 3: E8 мёртв из-за alpha + init
**Действие:** alpha 0.001→0.01, init_scale 0.01→0.1
**Ожидание:** controller реально адаптируется, E8 > E6

### Гипотеза 4: Lyapunov penalty слишком агрессивен
**Действие:** penalty_weight 0.5→0.15 (между оригинальным 0.1 и текущим 0.5)
**Ожидание:** E7 ≥ E6 (was E7 < E6)

### Приоритет выполнения
1. [CRITICAL] lambda_va rebalance — потенциально самый большой эффект
2. [HIGH] rho=0.3 — давно запланировано, не применено
3. [MEDIUM] E8 controller fix
4. [LOW] Lyapunov tuning

### Отложено
- Family B variants E8b-E10b (A > B доказано)
- Full training (10 эпох) — после балансировки лосса
- Fix rule activation logging (добавить log на firing steps)
