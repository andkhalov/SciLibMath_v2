# EXP-011: Sweep13 — Full Results & Analysis

> Дата: 2026-03-17
> Статус: завершён (32/32 ok, 0 failures)
> Время: 01:39–17:41 MSK (~16 часов)
> GPU: RTX 3090, 10% data (97,271 train / 4,863 test), 5 epochs, seed=42
> Checkpoints: DISABLED

---

## 1. Финальная таблица (sorted by cm@1)

```
 #  Experiment                   Fam    cm@1   cm@10  D_intra  img→c   en→c   ru→c  lean→c  ltx→c
─── ──────────────────────────── ──── ─────── ─────── ─────── ─────── ────── ────── ────── ──────
  1 e1_pairwise_cnxt             CNXT  0.4727  0.7033   1.219  0.825  0.891  0.909  0.841  0.913
  2 e1_pairwise                  A     0.4422  0.6572   1.357  0.786  0.890  0.901  0.841  0.912
  3 e1b_pairwise                 B     0.4213  0.6326   1.123  0.705  0.918  0.929  0.858  0.871
  4 e8c_low_va                   A     0.4140  0.6140   0.040  0.453  0.890  0.898  0.889  0.898
  5 e8c_pairwise                 A     0.4097  0.6165   0.042  0.311  0.854  0.860  0.873  0.865
  6 h51_wmin05                   H     0.4049  0.5976   0.041  0.360  0.884  0.894  0.890  0.894
  7 h53_combined                 H     0.4049  0.5965   0.041  0.375  0.890  0.893  0.891  0.895
  8 h53_combined_06              H     0.4047  0.5960   0.041  0.366  0.888  0.895  0.891  0.893
  9 e8c_active                   A     0.4043  0.5967   0.041  0.375  0.891  0.894  0.891  0.894
 10 h51_wmin06                   H     0.4042  0.5967   0.041  0.365  0.890  0.895  0.889  0.893
 11 h55_img_warmup               H     0.4028  0.5937   0.041  0.361  0.890  0.895  0.889  0.893
 12 e8c_active_cnxt              CNXT  0.3559  0.5450   0.017  0.945  0.978  0.986  0.967  0.951
 13 e8c_low_va_cnxt              CNXT  0.3525  0.5426   0.018  0.945  0.977  0.985  0.967  0.950
 14 e7_lyapunov                  A     0.3480  0.4903   0.041  0.134  0.897  0.897  0.910  0.884
 15 e6_fuzzy                     A     0.3466  0.4896   0.041  0.124  0.896  0.895  0.908  0.885
 16 e10_potential_fuzzy           A     0.3392  0.4868   0.026  0.127  0.901  0.899  0.906  0.885
 17 e9_potential                  A     0.3294  0.4839   0.027  0.948  0.984  0.991  0.967  0.950
 18 e6_low_elastic                A     0.3272  0.4750   0.043  0.652  0.935  0.948  0.856  0.814
 19 e7_lyapunov_cnxt             CNXT  0.3271  0.4763   0.047  0.169  0.892  0.893  0.906  0.883
 20 h52_low_align_e4             H     0.3194  0.4749   0.016  0.947  0.983  0.991  0.970  0.954
 21 e4_composite_static          A     0.3177  0.4734   0.016  0.945  0.981  0.990  0.974  0.953
 22 e5_composite_learnable       A     0.3171  0.4742   0.015  0.947  0.986  0.993  0.967  0.950
 23 h52_low_align                H     0.3151  0.4716   0.016  0.946  0.985  0.993  0.968  0.951
 24 e3_centroid_reg              A     0.3143  0.4706   0.015  0.947  0.984  0.992  0.970  0.953
 25 e9_potential_cnxt            CNXT  0.3099  0.4740   0.028  0.944  0.983  0.989  0.966  0.950
 26 e4b_composite_static         B     0.3074  0.4487   0.014  0.942  0.975  0.983  0.958  0.947
 27 e3b_centroid_reg             B     0.3055  0.4470   0.013  0.946  0.978  0.987  0.951  0.940
 28 e4_composite_static_cnxt     CNXT  0.3029  0.4662   0.017  0.940  0.982  0.991  0.972  0.954
 29 e5_composite_learnable_cnxt  CNXT  0.3005  0.4641   0.017  0.946  0.985  0.991  0.968  0.952
 30 e3_centroid_reg_cnxt         CNXT  0.2876  0.4545   0.018  0.938  0.986  0.991  0.969  0.950
 31 e10_potential_fuzzy_cnxt     CNXT  0.2789  0.4399   0.033  0.241  0.887  0.900  0.909  0.886
 32 e6_fuzzy_cnxt                CNXT  0.2709  0.4329   0.062  0.225  0.876  0.886  0.913  0.892
```

---

## 2. ConvNeXt-Pico vs ResNet18 (matched pairs)

```
Experiment               RN cm@1   CX cm@1    delta   RN img→c  CX img→c
─────────────────────── ──────── ──────── ──────── ──────── ────────
e1_pairwise              0.4422    0.4727   +0.0305    0.786    0.825
e3_centroid_reg          0.3143    0.2876   -0.0267    0.947    0.938
e4_composite_static      0.3177    0.3029   -0.0148    0.945    0.940
e5_composite_learnable   0.3171    0.3005   -0.0166    0.947    0.946
e6_fuzzy                 0.3466    0.2709   -0.0757    0.124    0.225
e7_lyapunov              0.3480    0.3271   -0.0209    0.134    0.169
e8c_active               0.4043    0.3559   -0.0484    0.375    0.945
e9_potential             0.3294    0.3099   -0.0195    0.948    0.944
e10_potential_fuzzy      0.3392    0.2789   -0.0603    0.127    0.241
e8c_low_va               0.4140    0.3525   -0.0615    0.453    0.945
─────────────────────────────────────────────────────────────────────
AVERAGE                                     -0.0314             +0.133
```

### Вывод по ConvNeXt

**Для pairwise (E1): ConvNeXt = чистый выигрыш.** +3.05pp cm@1, +3.9pp img→c R@1.
Это ожидаемо — лучший visual encoder → лучшие image embeddings → лучший cross-modal retrieval.

**Для centroid-based + controller: ConvNeXt = парадоксальный проигрыш.**
Среднее падение -3.14pp cm@1, несмотря на рост img→c. Причина:

1. ConvNeXt выдаёт сильные img embeddings → alignment loss коллапсирует все модальности к img
2. D_intra для CNXT-centroid = 0.017 vs 0.041 для ResNet → более жёсткий alignment collapse
3. Текстовые модальности теряют индивидуальность → cross-modal recall падает

**E8c_active_cnxt — ключевой пример парадокса:**
- img→c: 0.375 → **0.945** (+57pp!) — ConvNeXt полностью исправил image modality
- cm@1: 0.4043 → **0.3559** (−4.84pp) — но cross-modal retrieval стал хуже!

**Вывод:** ConvNeXt полезен ТОЛЬКО для pairwise loss, где нет alignment loss, толкающего к коллапсу. Для composite/controller loss нужен адаптивный alignment или modality-specific scaling.

---

## 3. H-гипотезы (sweep13 additions)

| Hypothesis | Experiment | cm@1 | vs baseline | img→c | Verdict |
|---|---|---|---|---|---|
| H51 (w_min=0.5) | h51_wmin05 | 0.4049 | +0.0006 vs e8c | 0.360 | neutral |
| H51 (w_min=0.6) | h51_wmin06 | 0.4042 | -0.0001 vs e8c | 0.365 | neutral |
| H52 (low align, E3) | h52_low_align | 0.3151 | +0.0008 vs e3 | 0.946 | neutral |
| H52 (low align, E4) | h52_low_align_e4 | 0.3194 | +0.0017 vs e4 | 0.947 | neutral |
| H53 (w_min+low_align) | h53_combined | 0.4049 | +0.0006 vs e8c | 0.375 | neutral |
| H53 (w_min=0.6+low_align) | h53_combined_06 | 0.4047 | +0.0004 vs e8c | 0.366 | neutral |
| H55 (img warmup) | h55_img_warmup | 0.4028 | -0.0015 vs e8c | 0.361 | slightly worse |

### Вывод по H-гипотезам

**Все нейтральны.** E8c nonlinear controller уже консервативен — w_min clamp и reduced alignment не дают эффекта потому что:

1. **H51 (w_min):** E8c не понижает w_img ниже ~0.5 (MLP-consequent, не linear rule) → floor на 0.5/0.6 не активируется
2. **H52 (low align):** Для E3/E4 без контроллера, reduce align → не помогает, потому что проблема не в λ_align, а в отсутствии адаптивности
3. **H53 (combined):** Сумма двух нейтральных = нейтральное
4. **H55 (warmup):** 1000 steps warmup при 1443 steps/epoch = первая эпоха без alignment → не достаточно чтобы img "закрепился"

**Тупиковая ветка.** Дальнейшая настройка controller hyperparameters = diminishing returns. Прорыв — в другой плоскости.

---

## 4. Family B (2 shared encoders vs 5 separate)

| | Family A | Family B | Delta |
|---|---|---|---|
| E1 pairwise | 0.4422 | 0.4213 | -2.09pp |
| E3 centroid_reg | 0.3143 | 0.3055 | -0.88pp |
| E4 composite_static | 0.3177 | 0.3074 | -1.03pp |

Family B стабильно на 1-2pp хуже Family A. Shared encoder для 4 текстовых модальностей (en/ru/lean/latex) теряет специализацию. Для Paper A — достаточный аргумент.

---

## 5. Иерархия подходов (ключевой вывод)

```
TIER 1 (cm@1 > 0.42): Pairwise loss
  e1_pairwise_cnxt    0.4727   ← BEST OVERALL (ConvNeXt backbone)
  e1_pairwise          0.4422   ← BEST ResNet18
  e1b_pairwise         0.4213   ← Family B (shared encoder penalty)

TIER 2 (cm@1 ≈ 0.40-0.42): Centroid-based + Controller
  e8c_low_va           0.4140   ← BEST CENTROID-BASED
  e8c_pairwise         0.4097
  h51-h55 variants     ~0.4040  (neutral vs e8c_active)
  e8c_active           0.4043

TIER 3 (cm@1 ≈ 0.33-0.36): Fuzzy/Lyapunov
  e7_lyapunov          0.3480
  e6_fuzzy             0.3466

TIER 4 (cm@1 ≈ 0.31-0.33): Centroid loss (no controller)
  e4_composite_static  0.3177
  e5_composite_learnable 0.3171
  e3_centroid_reg      0.3143
```

**Key insight:** Pairwise loss > Centroid-based > Controller-tuned centroid. Более сложные подходы НЕ компенсируют простоту PairwiseInfoNCE. Однако centroid-based (Tier 2) = 10x дешевле при инференсе (1 centroid query vs M² pairwise comparisons).

---

## 6. img→centroid R@1 — модальный анализ

Три кластера по img→c:

```
HIGH img→c (>0.7):   E1 pairwise variants: 0.705-0.825
                      → Нет alignment loss, img свободен, D_intra > 1.0

MEDIUM img→c (0.3-0.5): E8c controller variants: 0.311-0.453
                         → Controller удерживает img, D_intra ≈ 0.04

LOW img→c (<0.2):     E6/E7/E10 fuzzy: 0.124-0.134
                       → Death spiral: w_img → min, img перестал учиться
```

**Текстовые модальности (en, ru, lean, latex):** Стабильно 0.84-0.99 во ВСЕХ экспериментах.
**Image — единственная проблемная модальность.** 4 текста = взаимная поддержка через центроид. 1 image = изолирован, нет кросс-modal scaffolding.

---

## 7. Кандидаты на полный прогон (100% данных, 10 epochs)

### Рекомендация: 3 эксперимента

| # | Experiment | Обоснование |
|---|---|---|
| **1** | **e1_pairwise_cnxt** | Best overall (0.4727). ConvNeXt + pairwise = optimal. Paper A/B baseline. |
| **2** | **e1_pairwise** | Best ResNet18 (0.4422). Ablation vs ConvNeXt. Дешевле визуальный backbone. |
| **3** | **e8c_low_va** | Best centroid-based (0.4140). 10x cheaper inference. Controller validation. Paper C candidate. |

### Опциональный 4-й:
| **4** | e1b_pairwise | Family B validation (0.4213). Нужен для Paper A (shared vs separate encoders). |

### НЕ запускать на полных данных:
- H51-H55: нейтральные, не дают прироста → тупиковая ветка
- ConvNeXt + centroid/controller: парадоксальный проигрыш → нужна архитектурная работа
- E6/E7 fuzzy/lyapunov без E8c: death spiral не решён

---

## 8. Открытые вопросы для следующей итерации

1. **ConvNeXt paradox:** Почему лучший img encoder ухудшает centroid retrieval? Гипотеза: нужен modality-specific alignment scaling (λ_align_img ≠ λ_align_text)
2. **Pairwise vs Centroid gap:** 5.8pp gap (0.4422 → 0.4140) = цена centroid geometry. Можно ли сократить?
3. **Image isolation:** 4 text modalities + 1 img = structural imbalance. Варианты: (a) увеличить w_img × 4, (b) image-specific contrastive head, (c) hierarchical centroid (text_centroid + img → final_centroid)
4. **Scale effects:** 10% data sweep ≠ 100% data. Relative rankings могут измениться при larger data.

---

*EXP-011 | Sweep13 full results | SciLibMath_v2 | 2026-03-17*
