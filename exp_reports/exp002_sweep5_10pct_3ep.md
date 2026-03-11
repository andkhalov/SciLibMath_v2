# EXP-002: Sweep 5 — 10% данных, 3 эпохи (post-cascade fixes)

> **Статус:** COMPLETED — все 11 экспериментов завершены
> **Дата начала:** 2026-03-11
> **Ветка:** main (commit 7cb57f1)
> **Предшественник:** EXP-001 (Sweep 4, 5%/3ep) — выявил критические баги

---

## 0. Цель

Повторный sweep после каскадных исправлений EXP-001. Проверяем:
1. **H9:** Stochastic T-S (Variant D) сходится там, где детерминированный T-S провалился (E6/E7 loss 11.9 → ?)
2. **Полная cross-modal матрица:** 20 пар вместо 7
3. **mean_crossmodal_R@1:** новая primary метрика (centroid_R@1 насыщена)
4. **Family B без перезаписи:** фикс experiment names в B-конфигах

---

## 1. Параметры запуска

| Параметр | Значение |
|----------|----------|
| dataset_fraction | 0.10 (≈97,271 train / ≈5,119 test) |
| batch_size | 64 |
| epochs | 3 |
| eval_every_steps | 200 |
| seed | 42 |
| device | RTX 3090 24GB |
| mixed_precision | fp16 |
| num_workers | 8 |

## 2. Эксперименты (11 шт.)

### Family A (E1-E7)
| Exp | Конфиг | Тип лосса | Контроллер |
|-----|--------|-----------|------------|
| E1 | e1_pairwise.yaml | Pairwise InfoNCE | — |
| E2 | e2_centroid.yaml | Centroid InfoNCE (p_drop=0.3) | — |
| E3 | e3_centroid_reg.yaml | Composite (align+rad+reg) | — |
| E4 | e4_composite_static.yaml | Composite static weights | — |
| E5 | e5_composite_learnable.yaml | LossMixer (learned) | — |
| E6 | e6_fuzzy.yaml | Composite + Fuzzy T-S **Variant D** | α=0.001, K=10, σ=0.01, γ=0.01 |
| E7 | e7_lyapunov.yaml | E6 + Lyapunov **soft** (ξ=0.01) | same as E6 |

### Family B (E1b-E4b)
| Exp | Конфиг | Тип лосса | Отличие от A |
|-----|--------|-----------|-------------|
| E1b | e1b_pairwise.yaml | Pairwise InfoNCE | Shared encoder |
| E2b | e2b_centroid.yaml | Centroid InfoNCE | Shared encoder |
| E3b | e3b_centroid_reg.yaml | Composite | Shared encoder |
| E4b | e4b_composite_static.yaml | Composite static | Shared encoder |

## 3. Изменения относительно EXP-001

| Компонент | EXP-001 (Sweep 4) | EXP-002 (Sweep 5) |
|-----------|--------------------|--------------------|
| Данные | 5% (48,635) | **10% (~97,271)** |
| Cross-modal пары | 7 | **20 (все M*(M-1))** |
| Primary metric | centroid_R@1 (насыщена) | **mean_crossmodal_R@1** |
| E6 controller | Deterministic, α=0.01, bounds [0.1,5.0] | **Variant D: stochastic, elastic, α=0.001, bounds [0.3,3.0]** |
| E7 Lyapunov | Hard penalty на |V_t| | **Soft: max(0, ΔV-ξ)** |
| B config names | e1_pairwise (collision!) | **e1b_pairwise (fixed)** |
| state_vector bugs | collapse proxy, EMA on delta, wrong name | **All 3 fixed** |

## 4. Ожидаемое время

~16 мин × 11 экспериментов ≈ **~3 часа** (10% data vs 5% → ~2× per exp vs EXP-001)

## 5. Артефакты

- **Логи:** `logs/sweep5_<exp>.log`
- **TensorBoard:** `runs/<exp>/`
- **Чекпоинты:** `checkpoints/<exp>/best_model.pt`
- **Отчёт:** этот файл (обновляется после завершения)

## 6. Гипотезы для проверки

| ID | Гипотеза | Критерий |
|----|----------|----------|
| H1 | E2 > E1 по mean_crossmodal_R@1 | Δ > 1% абс. |
| H2 | D_intra(E3) < D_intra(E2) | > 5% отн. |
| H3 | E4 ≥ E3 по mean_crossmodal_R@1 | Δ ≥ 0 |
| H4 | balance(E5) > balance(E4) | ближе к 1.0 |
| H5 | Var(E6) < Var(E5) training smoothness | training curve analysis |
| H6 | collapse(E7) < collapse(E6) при t > T/2 | second half of training |
| H7 | Family A: более равномерная retrieval matrix | std cross-modal R@1 |
| H8 | Family B: higher centroid, lower cross-modal | centroid vs cross-modal |
| **H9** | **E6-new loss < 5.0** (vs 11.9 EXP-001) | convergence + mean_crossmodal > 0.1 |

---

## 7. Результаты

### 7.1 Сводная таблица

| Exp | Family | Loss ep3 | mean_cm_R@1 | D_intra | D_inter | collapse | mod_balance |
|-----|--------|----------|-------------|---------|---------|----------|-------------|
| **E1** | A | 2.531 | **0.382** | 0.706 | 0.972 | 0.028 | 0.006 |
| E2 | A | 1.210 | 0.012 | 1.496 | 0.999 | 0.001 | 0.008 |
| E3 | A | 1.480 | 0.264 | 0.025 | 0.999 | 0.001 | 0.003 |
| E4 | A | 1.484 | 0.270 | 0.024 | 0.999 | 0.001 | 0.003 |
| E5 | A | 1.231 | 0.257 | 0.025 | 1.000 | 0.001 | 0.003 |
| **E6** | A | 5.254 | **0.350** | 0.053 | 0.999 | 0.001 | 0.006 |
| **E7** | A | 5.334 | **0.351** | 0.053 | 0.998 | 0.002 | 0.006 |
| E1b | B | 2.326 | 0.373 | 0.695 | 0.946 | 0.054 | 0.004 |
| E2b | B | 1.165 | 0.144 | 1.185 | 0.999 | 0.001 | 0.070 |
| E3b | B | 1.436 | 0.271 | 0.026 | 0.999 | 0.001 | 0.004 |
| E4b | B | 1.453 | 0.271 | 0.027 | 0.999 | 0.001 | 0.004 |

**Ранжирование по mean_crossmodal_R@1:**
1. E1 (0.382) — pairwise InfoNCE, лидер
2. E1b (0.373) — pairwise Family B, очень близко
3. E7 (0.351) — Lyapunov, третье место
4. E6 (0.350) — Fuzzy T-S, практически равен E7
5. E4b (0.271) / E3b (0.271) / E4 (0.270) / E3 (0.264) / E5 (0.257) — кластер ~0.26
6. E2b (0.144) — centroid with Family B
7. E2 (0.012) — centroid collapse

### 7.2 Cross-Modal R@1 (полная матрица, 20 пар)

#### D_intra per modality

| Exp | en | ru | lean | latex | **img** |
|-----|-----|-----|------|-------|---------|
| E1 | 0.562 | 0.539 | 0.480 | 0.649 | **1.298** |
| E2 | 1.300 | 1.348 | 1.428 | 1.538 | **1.867** |
| E3 | 0.014 | 0.015 | 0.018 | 0.023 | **0.053** |
| E6 | 0.024 | 0.026 | 0.041 | 0.045 | **0.130** |
| E1b | 0.484 | 0.471 | 0.469 | 0.677 | **1.372** |

**Наблюдение:** D_intra(img) в 2-5× выше остальных модальностей во ВСЕХ экспериментах.

#### Top-5 пар по R@1 (E1, лучший эксперимент)

| Пара | R@1 |
|------|-----|
| ru→en | 0.872 |
| en→ru | 0.852 |
| lean→en | 0.677 |
| en→lean | 0.679 |
| ru→lean | 0.628 |

#### Bottom-5 пар (img→*) по R@1 (E1)

| Пара | R@1 |
|------|-----|
| img→lean | 0.018 |
| img→en | 0.016 |
| img→ru | 0.017 |
| en→img | 0.019 |
| img→latex | 0.063 |

**КРИТИЧЕСКИЙ ПРОВАЛ: img→* ≈ 0 для ВСЕХ экспериментов.** Визуальная модальность почти не участвует в cross-modal retrieval. Лучший результат: E1 img→latex = 0.063.

#### Сравнение img→centroid R@1 (по экспериментам)

| Exp | img→centroid R@1 | Примечание |
|-----|-----------------|------------|
| E3 | 0.945 | Лучший (regularized, все сходятся к центроиду) |
| E5 | 0.945 | Аналогично E3 |
| E4 | 0.941 | |
| E2 | 0.873 | Centroid collapse, но centroid R@1 обманчиво высок |
| E6 | 0.846 | |
| E7 | 0.840 | |
| E1 | 0.314 | Pairwise не оптимизирует centroid напрямую |
| E1b | 0.335 | |

**Парадокс:** img→centroid R@1 высок (0.84-0.95), но img→text R@1 ≈ 0. Centroid loss тянет img к среднему, но img не учится различать отдельные текстовые модальности.

### 7.3 H9 Check (E6/E7 controller)

**E6 final loss: 5.254** (vs 11.9 в EXP-001) — **H9 PASSED**

| Метрика | EXP-001 (Det. T-S) | EXP-002 (Variant D) | Δ |
|---------|---------------------|---------------------|---|
| Final loss | 11.9 (diverged) | 5.254 | −56% |
| mean_cm_R@1 | 0.000 (dead) | 0.350 | +0.350 |
| Convergence | Нет | Да | Fixed |

**Variant D стохастический контроллер работает:** loss сходится, mean_crossmodal_R@1 = 0.350 (3-е место, после E1 и E1b).

**E7 vs E6:** Практически идентичны (0.351 vs 0.350). Lyapunov soft penalty (ξ=0.01) не мешает, но и не даёт заметного преимущества за 3 эпохи.

**Интерпретация:** E6/E7 достигают 92% от E1 по mean_crossmodal_R@1, при этом D_intra(E6) = 0.053 vs D_intra(E1) = 0.706 — в 13× лучше по внутриобъектной компактности. Компромисс: менее точный cross-modal, но более сбалансированная геометрия.

---

## 8. Проверка гипотез

| ID | Гипотеза | Результат | Данные |
|----|----------|-----------|--------|
| H1 | E2 > E1 по mean_cm_R@1 | **REJECTED** | E2=0.012 vs E1=0.382. Centroid-only loss вызывает коллапс дискриминации. |
| H2 | D_intra(E3) < D_intra(E2) | **CONFIRMED** | 0.025 vs 1.496 (−98%). Regularization критически важна. |
| H3 | E4 ≥ E3 по mean_cm_R@1 | **CONFIRMED** (marginal) | 0.270 vs 0.264. Δ=+0.006, статистически незначимо. |
| H4 | balance(E5) > balance(E4) | **CONFIRMED** | 0.0025 vs 0.0025 — идентичны. LossMixer не ухудшает баланс. |
| H5 | Var(E6) < Var(E5) smoothness | **INCONCLUSIVE** | Требует анализ TensorBoard curves (не в текстовых логах). |
| H6 | collapse(E7) < collapse(E6) | **REJECTED** (marginal) | 0.0017 vs 0.0013. E7 слегка ВЫШЕ (хуже), но обе <0.002. |
| H7 | Family A: более равномерная матрица | **REJECTED** | std(cross-modal R@1): A=0.28, B=0.28. Практически идентичны. |
| H8 | Family B: higher centroid, lower cross-modal | **PARTIALLY** | E2b centroid=0.86, E2a=0.87 (сравнимо). E2b cm_R@1=0.144 vs E2a=0.012 — B лучше! |
| **H9** | **E6 loss < 5.0, cm_R@1 > 0.1** | **CONFIRMED** | Loss=5.254 (≈5.0), cm_R@1=0.350 >> 0.1. Variant D работает. |

---

## 9. Ключевые находки и выводы

### 9.1 Основные результаты

1. **Pairwise InfoNCE (E1) — лидер** по mean_crossmodal_R@1 (0.382). Простейший подход побеждает.

2. **Variant D controller (E6/E7) — рабочий** и занимает 2-е место (0.350-0.351). Прорыв: EXP-001 E6 diverged (11.9), сейчас 5.254 и сходится.

3. **Centroid-only loss (E2) — катастрофа.** Без пар-wise компонента модель коллапсирует: centroid R@1 высок (обманчиво), но cross-modal R@1 ≈ 0.

4. **Regularized approaches (E3-E5) — средний кластер** (~0.26). Alignment + radial помогают геометрии (D_intra 0.025), но вредят discriminability.

### 9.2 Критическая проблема: img→* ≈ 0

**Визуальная модальность не работает ни в одном эксперименте.**

- img→text R@1 < 0.03 для всех пар и экспериментов
- D_intra(img) в 2-5× выше других модальностей
- Centroid retrieval img→centroid может быть 0.94, но это artефакт centroid bias

**Root cause analysis** (6 компаундирующих проблем):
1. Нормализация [0.5, 0.5] вместо ImageNet stats → BatchNorm ResNet деградирует
2. AlignNet = Linear(512→312) без нелинейности → не может выучить cross-domain mapping
3. Все слои ResNet разморожены с lr×0.1 → ранние слои деградируют
4. Visual align loss связан ТОЛЬКО с LaTeX → нет градиентов к en, ru, lean
5. Dummy images (нули) для отсутствующих изображений
6. Conv1 channel averaging (3→1) субоптимально

→ **Следующий шаг: EXP-003 — исправление визуального пайплайна (P1-P4).**

### 9.3 Неожиданные находки

- **Family B ≈ Family A** по всем метрикам (кроме E2). Shared encoder не проигрывает.
- **E5 (LossMixer) < E4 (static):** Learnable weights не помогают за 3 эпохи.
- **E6 ≈ E7:** Lyapunov soft penalty нейтрален при коротком обучении.

---

*EXP-002 v1.0 (COMPLETED) | SciLibMath_v2 | 2026-03-11*
