# EXP-010: Sweep13 — H-Hypotheses + Full Rerun (Plan)

> Дата: 2026-03-17
> Статус: подготовка завершена, ожидание запуска
> Предыдущий sweep: EXP-009 (sweep12, 22/25 ok, 1 fail, 2 missing)

---

## 1. Итоги sweep12 (EXP-009)

### 1.1 Статус
- **22/25 экспериментов** завершены успешно
- **1 fail:** `e9_potential_cnxt` — disk I/O error на epoch 4/5 (checkpoint save)
- **2 missing:** `e10_potential_fuzzy_cnxt`, `e8c_low_va_cnxt` — не запустились (после crash)
- **e9_potential_cnxt:** 3 эпохи завершены, best_model сохранён, eval данные в TensorBoard

### 1.2 Причина падения
- Чекпоинты: 109GB за 24 эксперимента (~4.5GB каждый × keep_best=3 × 25)
- Диск: 90% usage (78GB свободно) → I/O error при сохранении epoch checkpoint
- **Решение:** `checkpoint.enabled=false` в sweep13, чекпоинты не нужны для ablation sweep

### 1.3 Финальная таблица sweep12 (cm_R@1, отсортировано)

| # | Experiment | Family | cm_R@1 | cm_R@10 | D_intra | img→c R@1 | HM |
|---|-----------|--------|--------|---------|---------|-----------|------|
| 1 | e1_pairwise_cnxt | A/CNXT | **0.4720** | 0.7035 | 1.212 | 0.841 | **0.614** |
| 2 | e1_pairwise | A | 0.4429 | 0.659 | 1.283 | 0.773 | 0.586 |
| 3 | e1b_pairwise | B | 0.4242 | 0.633 | 1.069 | 0.699 | 0.568 |
| 4 | e8c_low_va | A | 0.4136 | 0.618 | 0.040 | 0.436 | 0.546 |
| 5 | e8c_active | A | 0.4059 | 0.599 | 0.041 | 0.374 | 0.536 |
| 6 | e8c_pairwise | A | 0.4079 | 0.604 | 0.037 | 0.309 | 0.529 |
| 7 | e9_potential | A | 0.3260 | 0.486 | 0.027 | 0.946 | 0.488 |
| 8 | e4_composite_static | A | 0.3184 | 0.479 | 0.015 | 0.944 | 0.479 |
| 9 | e3_centroid_reg | A | 0.3139 | 0.474 | 0.015 | 0.944 | 0.474 |
| 10 | e6_fuzzy | A | 0.3479 | 0.496 | 0.040 | 0.125 | 0.474 |
| 11 | e6_low_elastic | A | 0.3309 | 0.487 | 0.040 | 0.583 | 0.473 |
| 12 | e10_potential_fuzzy | A | 0.3388 | 0.489 | 0.026 | 0.140 | 0.466 |
| 13 | e7_lyapunov | A | 0.3479 | 0.497 | 0.040 | 0.133 | 0.464 |
| 14 | e5_composite_learnable | A | 0.3137 | 0.467 | 0.015 | 0.937 | 0.472 |
| 15 | e3b_centroid_reg | B | 0.2818 | 0.430 | 0.015 | 0.970 | 0.430 |
| 16 | e4b_composite_static | B | 0.2829 | 0.436 | 0.011 | 0.956 | 0.436 |
| 17 | e3_centroid_reg_cnxt | A/CNXT | 0.2982 | 0.457 | 0.019 | 0.949 | 0.454 |
| 18 | e4_composite_static_cnxt | A/CNXT | 0.2773 | 0.443 | 0.017 | 0.946 | 0.426 |
| 19 | e5_composite_learnable_cnxt | A/CNXT | 0.2844 | 0.436 | 0.017 | 0.945 | 0.437 |
| 20 | e6_fuzzy_cnxt | A/CNXT | 0.3361 | 0.483 | 0.043 | 0.152 | 0.456 |
| 21 | e7_lyapunov_cnxt | A/CNXT | 0.3284 | 0.489 | 0.044 | 0.150 | 0.449 |
| 22 | e8c_active_cnxt | A/CNXT | 0.3574 | 0.549 | 0.017 | 0.945 | 0.517 |
| — | e9_potential_cnxt | A/CNXT | ~0.30* | — | — | — | — |
| — | e10_potential_fuzzy_cnxt | A/CNXT | MISSING | — | — | — | — |
| — | e8c_low_va_cnxt | A/CNXT | MISSING | — | — | — | — |

*e9_potential_cnxt: 3/5 epochs completed, approximate from partial data.

---

## 2. Корневые проблемы (из диагностики sweep12)

1. **Alignment-collapse tradeoff:** L_align тянет e_m→centroid, при D_intra→0 модальности неразличимы
2. **Controller death spiral (E6):** w_img: 1.0→0.3, img перестаёт учиться
3. **Text centroid bias:** c = (4×text + 1×img)/5, img = 20% влияния

---

## 3. Sweep13: план и гипотезы

### 3.1 Новые гипотезы

| ID | Гипотеза | Конфиг | Ожидание |
|----|----------|--------|----------|
| **H51** | w_min floor предотвращает death spiral | h51_wmin05 (0.5), h51_wmin06 (0.6) | img→c R@1 > 0.5 (was 0.12 in E6) |
| **H52** | λ_align÷10 сохраняет модальную структуру | h52_low_align (E3), h52_low_align_e4 (E4) | cm_R@1 > 0.35 при D_intra > 0.1 |
| **H53** | H51+H52 combined = лучший controller-based | h53_combined (0.5), h53_combined_06 (0.6) | cm_R@1 > 0.42, img→c > 0.5 |
| **H55** | Alignment warmup даёт img время "войти" | h55_img_warmup (1000 steps) | img→c R@1 > E8c (0.37) |

### 3.2 Имплементация

- **H51 (w_min):** `project_to_bounds()` + `elastic_step()` + `TSFuzzyController.__init__()` — параметр `w_min` overrides lower bound для indices 5-10
- **H52 (low align):** Чисто конфиг: `lambda_align: 0.03`
- **H55 (warmup):** `CompositeLoss.align_warmup_steps` — `align_scale=0` during warmup, passed via `set_step()`

### 3.3 Полный список экспериментов (32)

```
Family A (12):    e1_pairwise, e3_centroid_reg, e4_composite_static,
                  e5_composite_learnable, e6_fuzzy, e7_lyapunov,
                  e8c_active, e9_potential, e10_potential_fuzzy,
                  e8c_low_va, e8c_pairwise, e6_low_elastic

Family B (3):     e1b_pairwise, e3b_centroid_reg, e4b_composite_static

ConvNeXt (10):    e1_pairwise_cnxt, e3_centroid_reg_cnxt,
                  e4_composite_static_cnxt, e5_composite_learnable_cnxt,
                  e6_fuzzy_cnxt, e7_lyapunov_cnxt, e8c_active_cnxt,
                  e9_potential_cnxt, e10_potential_fuzzy_cnxt, e8c_low_va_cnxt

H-hypotheses (7): h51_wmin05, h51_wmin06, h52_low_align, h52_low_align_e4,
                  h53_combined, h53_combined_06, h55_img_warmup
```

### 3.4 Параметры

- **Data:** 10% (92,408 train / 4,863 test), seed=42
- **Training:** 5 epochs, batch_size=64, lr=1e-4, gradient_clip=1.0
- **Checkpoints:** DISABLED (checkpoint.enabled=false)
- **Оценка:** eval_every_steps=200
- **Метрики:** cm_R@1, cm_R@10, D_intra, D_inter, collapse, mod_balance, img→c R@1
- **Estimated time:** ~16h (32 × ~30min each)

### 3.5 Зачем полный rerun?

Sweep12 был прерван (disk error), 2 эксперимента missing. Sweep13 в одинаковых условиях
(10%, 5ep, same GPU, same seed, no checkpoints) даёт чистый сравнительный срез.
Все 32 эксперимента в одном sweep = нет bias от разных run conditions.

---

## 4. Изменения кода

| Файл | Изменение |
|------|-----------|
| `code/train.py` | checkpoint guard (`cfg.checkpoint.enabled`), H-experiments in dispatch/controller lists, `set_step()` call for H55 |
| `code/controller/rules.py` | `w_min` param in `project_to_bounds()` and `elastic_step()` |
| `code/controller/ts_controller.py` | `w_min` param in `__init__()`, passed to `elastic_step()` |
| `code/losses/composite.py` | `align_warmup_steps`, `set_step()`, `align_scale` in forward |
| `configs/base.yaml` | `checkpoint.enabled: true` (default) |
| `run_ablation.sh` | +7 H-experiments, `checkpoint.enabled=false` in overrides |
| `tests/test_sweep13_features.py` | 10 tests for H51, H55, checkpoint disable |

---

*EXP-010 plan | SciLibMath_v2 | 2026-03-17*
