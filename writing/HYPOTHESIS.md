# HYPOTHESIS.md — Тестируемые гипотезы SciLibMath_v2

> **Версия:** 1.0 | **Дата:** 2026-03-10
> **Привязка:** MATH.md v2.6.2, TZ.md v1.6
> **Sweep:** 5% данных, 3 эпохи, seed=42

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

## Результаты (заполняется после sweep)

_Placeholder: результаты будут добавлены после завершения sweep 5%/3ep._

| ID | Подтверждена? | Ключевые числа | Комментарий |
|----|---------------|----------------|-------------|
| H1 | — | — | — |
| H2 | — | — | — |
| H3 | — | — | — |
| H4 | — | — | — |
| H5 | — | — | — |
| H6 | — | — | — |
| H7 | — | — | — |
| H8 | — | — | — |
| T3 | — | — | — |
| T7 | — | — | — |
| T9 | — | — | — |

---

*HYPOTHESIS.md v1.0 | SciLibMath_v2 | 2026-03-10*
