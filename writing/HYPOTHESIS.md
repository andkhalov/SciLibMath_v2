# HYPOTHESIS.md — Тестируемые гипотезы SciLibMath_v2

> **Версия:** 2.0 | **Дата:** 2026-03-15
> **Привязка:** MATH.md v2.7.0, TZ.md v1.7
> **Данные:** sweep4-10 (5%-20%, 3-5 эпох, seed=42)

---

## Назначение

Этот файл фиксирует тестируемые гипотезы **до** запуска sweep.
Каждая гипотеза привязана к конкретным экспериментам, метрикам и секциям формализации.
Результаты заполняются после sweep.

---

## Таблица гипотез

### Основные гипотезы (H1-H8)

| ID | Гипотеза | Тест | Метрика | Ожидание | Ref (MATH.md / TZ.md) |
|----|----------|------|---------|----------|------------------------|
| **H1** | Centroid InfoNCE превосходит Pairwise для M>2 модальностей | E2 vs E1 | mean_crossmodal_R@1 | E2 > E1 | M.3.2, T.7, T.9 |
| **H2** | Alignment + Radial regularization улучшают геометрию | E3 vs E2 | D_intra ↓, D_inter ↑ | D_intra(E3) < D_intra(E2), D_inter(E3) > D_inter(E2) | M.3.3 |
| **H3** | Composite loss с per-modality компонентами ≥ Centroid | E4 vs E3 | mean_crossmodal_R@1, collapse_score | E4 ≥ E3, collapse(E4) ≤ collapse(E3) | M.3.4 |
| **H4** | Learnable weights (LossMixer) адаптивнее static weights | E5 vs E4 | modality_balance | balance(E5) > balance(E4) | M.3.5 |
| **H5** | Fuzzy T-S controller (Variant D) стабильнее LossMixer | E6 vs E5 | Var(mean_crossmodal_R@1 across evals), training smoothness | Var(E6) < Var(E5) | M.6, M.6.3b, T.6 |
| **H6** | Lyapunov soft constraint предотвращает поздний коллапс | E7 vs E6 | collapse_score на последних eval, V_t trajectory | collapse(E7) < collapse(E6) при t > T/2 | M.7, M.7.2b, T.5 |
| **H7** | Family A лучше на cross-modal семантику (разнообразие пар) | A(E1-E4) vs B(E1b-E4b) | retrieval_matrix diversity (std cross-modal R@1) | A: более равномерная матрица | M.1.1 vs M.1.3 |
| **H8** | Family B выше centroid R@1 но хуже cross-modal | A(E1-E4) vs B(E1b-E4b) | centroid_R@1, cross_modal_R@1 | B: higher centroid, lower cross-modal | M.1.3 |

### H9 — Стохастический T-S контроллер (Variant D)

| ID | Гипотеза | Тест | Метрика | Ожидание | Ref |
|----|----------|------|---------|----------|-----|
| **H9** | Stochastic T-S (Variant D) сходится там, где детерминированный T-S fail | E6-new vs E6-old | loss convergence + mean_crossmodal_R@1 | E6-new loss < 5.0 (vs 11.9 for E6-old); mean_crossmodal_R@1(E6-new) > 0.1 | M.6.3b |

**[EXP-001 Context]:** Детерминированный T-S контроллер (original M.6.3) crashed к corners: все w_m → MIN, w_g → MAX, loss=11.9 (vs 2.5 baseline E4). H9 тестирует, решает ли Variant D (elastic reversion + stochastic noise + narrow bounds) эту проблему.

### Теоретические предсказания (T3, T7, T9)

| ID | Предсказание | Мониторинг | Метрика | Ожидание | Ref |
|----|-------------|------------|---------|----------|-----|
| **T3** | Retrieval guarantee: δ > 4Mε обеспечивает centroid retrieval | Все E1-E7 | D_inter / (4 * M * sqrt(D_intra)) | > 1.0 для успешного retrieval | M.9, T.3 |
| **T7** | Centroid loss покрывает все модальности, pairwise — только 2/M | E1 vs E2 | gradient signal per modality (weight_norm change rate) | E2: все модальности обучаются; E1: неравномерно | M.9, T.7 |
| **T9** | Centroid обобщает лучше (меньший train-val gap) | E1 vs E2 | train_loss - eval_loss (proxy: centroid_R@1 стабильность) | gap(E2) / gap(E1) ≈ 1/M | M.9, T.9 |

---

## Протокол оценки

### Метрики для сравнения

1. **mean_crossmodal_R@{1,3,10}** — **PRIMARY METRIC** (среднее R@k по всем 20 направленным парам модальностей)
2. **centroid_R@{1,3,10}** — LOO centroid retrieval (NOTE: structurally saturates at ~1.0 due to 4/5 overlap; use as sanity check, not primary metric)
3. **D_intra** — средний радиус объекта (меньше = лучше)
4. **D_inter** — среднее расстояние между центроидами (больше = лучше)
5. **collapse_score** — индикатор коллапса (меньше = лучше)
6. **modality_balance** — баланс модальностей (ближе к 1.0 = лучше)

### Условие принятия гипотезы

- **Количественное:** разница метрик > 1% абсолютных для R@k, или > 5% относительных для geometry
- **Качественное:** направление изменения соответствует ожиданию
- **Caveat:** При 5%/3ep и single seed — это предварительные результаты; окончательное подтверждение требует full training с multiple seeds

### Визуализации

- **retrieval_matrix/R@1** — тепловая карта 5x5 (TensorBoard)
- **retrieval_matrix/R@10** — тепловая карта 5x5 (TensorBoard)
- **pca/simplex_5objects** — PCA проекция 5 объектов с центроидами (TensorBoard)
- **loss/** — компоненты лосса (TensorBoard scalars)
- **controller/** — состояние fuzzy controller для E6-E7 (TensorBoard scalars)

---

## Результаты H1-H9 (кумулятивные по sweep4-10)

> Данные: 6 sweep'ов, 5%-20% данных, 3-5 эпох. Финальные числа из sweep10 (20%, 5ep, live controller).

| ID | Подтверждена? | Ключевые числа | Комментарий |
|----|---------------|----------------|-------------|
| H1 | **ОТВЕРГНУТА** | E2=0.092 << E1=0.471 | Centroid InfoNCE без регуляризации = катастрофа. Stable across 6 sweeps. |
| H2 | **ПОДТВЕРЖДЕНА** | D_intra: E3=0.004 << E2=1.546 | Alignment+radial стабилизируют геометрию. Но E3<E1 по R@1. |
| H3 | **ПОДТВЕРЖДЕНА** | E4=0.356 > E3=0.353 (+0.3pp) | Marginal improvement. Per-modality weights (lean/latex=1.5) помогают. |
| H4 | **ОТВЕРГНУТА** | E5=0.353 ≈ E4=0.356 | LossMixer не улучшает. Gradient-based weights не видят collapse signals. |
| H5 | **ПОДТВЕРЖДЕНА** | E6=0.381 > E5=0.353 (+2.8pp) | Fuzzy > LossMixer. Controller видит state signals (даже с малым drift). |
| H6 | **ОТВЕРГНУТА** | E7=0.380 ≈ E6=0.381 | Lyapunov penalty не даёт пользы. V_t penalty слишком мал при α=0.001. |
| H7 | **ПОДТВЕРЖДЕНА** | A>B: +2.4pp (E1), +3.3pp (E3), +3.5pp (E4) | Separate encoders лучше. Gap в latex↔lean (-12pp у Family B). |
| H8 | **ОТВЕРГНУТА** | B хуже и по centroid, и по cross-modal | Family B хуже во всём. Shared encoder теряет модальную специфику. |
| H9 | **ПОДТВЕРЖДЕНА** | E6 loss=2.3 (vs 11.9 old), cm_R@1=0.381 | Variant D стабилен. Детерминированный T-S crashed (EXP-001). |
| T3 | **Не применимо** | D_inter/D_intra ratio varies | Нужен полный train для надёжной проверки |
| T7 | **ЧАСТИЧНО** | E1 img backbone norm >> text | Pairwise E1 обучает img сильнее (выше weight_norm). Centroid — равномернее. |
| T9 | **Не применимо** | — | Нужен полный train с отдельным val set |

---

### Гипотезы EXP-007: H31-H40 (sweep10, live controller)

> Зафиксированы до sweep10, проверены после. Ref: DIARY.md 2026-03-14.

| ID | Гипотеза | Вердикт | Доказательство |
|----|----------|---------|----------------|
| **H31** | Живой E6 > мёртвый E6 (sweep7: 0.419) | **ОТВЕРГНУТА** | E6=0.381 < 0.419. α=0.001 слишком мал, drift ±0.001-0.008 |
| **H32** | E8c (nonlinear) > E6 (linear) | **ОТВЕРГНУТА** | E8c=0.376 < E6=0.381 |
| **H33** | E8c_low_va = best overall | **ОТВЕРГНУТА** | E8c_low_va=0.376, E1=0.471 best |
| **H34** | E7 > E6 при живом контроллере | **ОТВЕРГНУТА** | E7=0.380 ≈ E6=0.381 |
| **H35** | E9 > E3 по D_intra | **ПОДТВЕРЖДЕНА** | D_intra: E9=0.011 vs E3=0.004 (+175%) |
| **H36** | E10 > E9 | **ПОДТВЕРЖДЕНА** | E10=0.397 > E9=0.371 (+2.6pp) |
| **H37** | Family A > B на 2-4pp | **ПОДТВЕРЖДЕНА** | E1: +2.4pp, E3: +3.3pp, E4: +3.5pp |
| **H38** | Lambda trajectory нестационарна | **ЧАСТИЧНО** | Нестационарна, но drift ±0.001-0.008 — слабо |
| **H39** | Rule activations > 0 после warmup | **ПОДТВЕРЖДЕНА** | 1444 controller events за sweep |
| **H40** | E5 < E6 | **ПОДТВЕРЖДЕНА** | E5=0.353 < E6=0.381 (+2.8pp) |

**Итог H31-H40:** 5 подтверждено, 4 отвергнуто, 1 частично.

---

### Гипотезы EXP-008: H41-H50 (sweep11, зафиксированы до запуска)

> Ref: MATH.md v2.8.0, EXP-008 plan
> NOTE: H41-H45 переформулированы — вместо OCR encoder используется ConvNeXt-Pico backbone (см. DIARY 2026-03-15 ~22:00)

| ID | Гипотеза | Метрика | Ожидание | Ref |
|----|----------|---------|----------|-----|
| **H41** | ConvNeXt-Pico backbone улучшает img→text retrieval vs ResNet18 | img→lat R@1 | E1_cnxt > E1 (0.39) | M.1.2c |
| **H42** | ConvNeXt backbone улучшает cm_R@1 в pairwise | cm_R@1 | E1_cnxt > E1 (0.47) | M.1.2c |
| **H43** | ConvNeXt backbone улучшает centroid-based | cm_R@1 | E10_cnxt > E10 (0.40) | M.1.2c |
| **H44** | ConvNeXt + controller > ConvNeXt без controller | cm_R@1 | E6_cnxt > E3_cnxt | M.6, M.1.2c |
| **H45** | ConvNeXt эффект сильнее для centroid чем pairwise | Δcm_R@1 | (E10_cnxt-E10) > (E1_cnxt-E1) | M.1.2c |
| **H46** | L_va curriculum снижает VA доминирование | L_va/L_total step 5K | < 50% (было 79%) | M.2.4 |
| **H47** | α×5 / γ÷5 усиливает контроллер | lambda drift range | > ±0.05 (было ±0.008) | M.6.3b |
| **H48** | α×5 / γ÷5 улучшает E6 vs E4 | E6-E4 cm_R@1 | > +0.02 (было +0.025) | M.6.3b |
| **H49** | final_div_factor=10 убирает weight norm plateau | weight_norm last step | > weight_norm at step 10K | — |
| **H50** | Combined fixes (B+C+D) улучшают centroid-based | E10 cm_R@1 | > 0.42 (было 0.397) | — |

#### Результаты H41-H50 (sweep11, 10% данных, 5 эпох, 23 эксперимента)

> Sweep11 завершён 2026-03-16 10:06 MSK, 23/23 OK. Ref: DIARY.md 2026-03-16 11:30.

| ID | Вердикт | Ключевые числа | Комментарий |
|----|---------|----------------|-------------|
| **H41** | **ПОДТВЕРЖДЕНА** | img→lat: E1_cnxt=0.3364 vs E1=0.1978 (+70%) | ConvNeXt даёт лучшие визуальные фичи для img→text |
| **H42** | **ПОДТВЕРЖДЕНА** | E1_cnxt=0.4783 > E1=0.4430 (+3.5pp) | Новый рекорд проекта. Превзошёл sweep10 E1=0.4705 на 10% данных |
| **H43** | **ОТВЕРГНУТА** | E10_cnxt=0.3043 < E10=0.3412 (-3.7pp) | ConvNeXt ХУЖЕ для centroid-based. Регуляризация overconstrains CNXT embeddings (D_intra=0.030 vs 0.026) |
| **H44** | **ПОДТВЕРЖДЕНА** | E6_cnxt=0.3361 > E3_cnxt=0.2982 (+3.8pp) | Controller помогает и с ConvNeXt backbone |
| **H45** | **ОТВЕРГНУТА** | Δ_pairwise=+0.035, Δ_centroid=-0.037 | Обратный эффект: CNXT помогает pairwise, вредит centroid. Парадокс — идеальное img→c alignment (0.94) ухудшает cm_R@1 |
| **H46** | **ЧАСТИЧНО** | Нет прямых L_va/L_total в логах | Косвенно: E8c_low_va скачок +4pp vs sweep10 при меньших данных |
| **H47** | **ЧАСТИЧНО** | Нет lambda drift в stdout логах | Косвенно: E8c #7→#3 в ранке, E6-E4 gap вырос. Нужен TensorBoard |
| **H48** | **ПОДТВЕРЖДЕНА** | E6-E4=+0.035 (было +0.025 в sweep10) | Gap увеличился на 40% — controller стал эффективнее |
| **H49** | **НЕ ПРОВЕРЕНА** | weight_norm не в stdout логах | Требуется анализ TensorBoard runs |
| **H50** | **ОТВЕРГНУТА** | E10=0.3412 < 0.42 | E10 не достиг 0.42. Но E8c_low_va=0.4159 ≈ 0.42 — nonlinear controller выиграл вместо potential+fuzzy |

**Итог H41-H50:** 4 подтверждены, 3 отвергнуты, 2 частично, 1 не проверена.

**Ключевой неожиданный результат:** E8c (nonlinear MLP consequents) — главный бенефициар EXP-008.
Скачок с rank #7 на #3, gap к E1 сократился с 9.5pp до 2.7pp. Гипотеза H50 формулировала
ожидание для E10, но выиграл E8c — нелинейные консеквенты + ослабленный elastic (γ÷5) оказались
важнее, чем потенциальные функции + fuzzy controller.

---

### Гипотезы EXP-010: H51-H55 (sweep13, зафиксированы до запуска)

> Ref: exp_reports/exp009_sweep12_interim_analysis.md (диагностика)
> Мотивация: sweep12 показал три корневые проблемы:
> (1) alignment-collapse tradeoff, (2) controller death spiral для img, (3) text centroid bias

| ID | Гипотеза | Конфиг | Метрика | Ожидание | Ref |
|----|----------|--------|---------|----------|-----|
| **H51** | w_min floor предотвращает death spiral: w_m ≥ 0.5 → img→c восстанавливается | h51_wmin05, h51_wmin06 | img→c R@1 | > 0.5 (was 0.12 in E6, 0.37 in E8c) | M.6.3 |
| **H52** | λ_align÷10 сохраняет модальную структуру при D_intra > 0.1 | h52_low_align (E3-based), h52_low_align_e4 (E4-based) | cm_R@1, D_intra | cm_R@1 > 0.35, D_intra > 0.1 (was 0.015) | M.3.3 |
| **H53** | w_min + reduced align (combined) = best controller-based | h53_combined, h53_combined_06 | cm_R@1 | > 0.42 (E8c_low_va=0.416) | M.6.3, M.3.3 |
| **H54** | Asymmetric centroid weighting (α_img=0.5) компенсирует text bias | — (deferred) | cm_R@1, img→lat R@1 | — | M.0 |
| **H55** | Alignment warmup (1000 steps) даёт img время войти в пространство | h55_img_warmup | img→c R@1 | > 0.37 (E8c baseline) | M.3.3 |

**H54 deferred:** Требует изменения модели (centroid computation), не конфига/контроллера.
Будет реализован в sweep14 если H51-H53 недостаточны.

---

*HYPOTHESIS.md v2.2 | SciLibMath_v2 | 2026-03-17*
