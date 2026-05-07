# EXP-008: ConvNeXt Backbone + Controller Tuning + Curriculum + LR Fix

> **Sweep:** sweep11 | **Data:** 10% (92,408 train / 4,863 test) | **Epochs:** 5
> **Date:** 2026-03-15 21:34 → 2026-03-16 10:06 MSK (~12.5h)
> **Result:** 23/23 OK, 0 failures

---

## 1. Мотивация

Sweep10 (EXP-007) показал: pairwise E1 = безусловный лидер (0.4705), контроллер E6 работает но drift мал (±0.008), Lyapunov инертен. EXP-008 вносит 4 изменения:

| ID | Изменение | Цель |
|----|-----------|------|
| A | ConvNeXt-Pico backbone (10 новых экспериментов) | Заменить ResNet18 на modern CNN |
| B | L_va curriculum: warmup=500, ramp=200 | Снизить VA доминирование (было 79% total loss) |
| C | α=0.005, γ=0.002 (было 0.001/0.01) | Усилить контроллер (α/γ: 0.1→2.5) |
| D | final_div_factor=10 (было 1e4) | Убрать LR plateau (lr→4e-10) |

Также: pivot OCR (pix2text-mfr) → ConvNeXt-Pico (OOM при DeiT-small в patch pipeline).

## 2. Дизайн

23 эксперимента = 10 Family A (ResNet18) + 3 Family B + 10 Family A (ConvNeXt-Pico).
Изменения B/C/D применены ко ВСЕМ экспериментам. Изменение A — только к 10 ConvNeXt.

## 3. Результаты

### 3.1 Полная таблица (sorted by cm_R@1)

| # | Experiment | Backbone | cm_R@1 | cm_R@10 | D_intra | img→c R@1 | img→lat R@1 |
|---|-----------|----------|--------|---------|---------|-----------|-------------|
| 1 | e1_pairwise_cnxt | CNXT | **0.4783** | 0.7090 | 1.2118 | 0.8390 | 0.3364 |
| 2 | e1_pairwise | ResNet | 0.4430 | 0.6590 | 1.2928 | 0.7744 | 0.1978 |
| 3 | e1b_pairwise | FamB | 0.4224 | 0.6327 | 1.0692 | 0.6726 | 0.1542 |
| 4 | **e8c_low_va** | ResNet | **0.4159** | 0.6176 | 0.0399 | 0.4530 | 0.1678 |
| 5 | e8c_active | ResNet | 0.4051 | 0.5987 | 0.0406 | 0.3632 | 0.1304 |
| 6 | e8c_low_va_cnxt | CNXT | 0.3601 | 0.5534 | 0.0173 | 0.9424 | — |
| 7 | e8c_active_cnxt | CNXT | 0.3574 | 0.5487 | 0.0173 | 0.9449 | — |
| 8 | e6_fuzzy | ResNet | 0.3517 | 0.4959 | 0.0400 | 0.1335 | 0.0113 |
| 9 | e7_lyapunov | ResNet | 0.3508 | 0.4972 | 0.0402 | 0.1330 | 0.0121 |
| 10 | e10_pot_fuzzy | ResNet | 0.3412 | 0.4891 | 0.0257 | 0.1337 | 0.0086 |
| 11 | e6_fuzzy_cnxt | CNXT | 0.3361 | 0.4834 | 0.0434 | 0.1520 | 0.0099 |
| 12 | e9_potential | ResNet | 0.3309 | 0.4857 | 0.0267 | 0.9472 | 0.0051 |
| 13 | e4_composite | ResNet | 0.3167 | 0.4730 | 0.0159 | 0.9472 | 0.0050 |
| 14 | e3_centroid_reg | ResNet | 0.3146 | 0.4711 | 0.0155 | 0.9478 | 0.0047 |
| 15 | e5_learnable | ResNet | 0.3146 | 0.4700 | 0.0155 | 0.9443 | 0.0044 |
| 16 | e9_pot_cnxt | CNXT | 0.3091 | 0.4743 | 0.0277 | 0.9445 | — |
| 17 | e7_lyap_cnxt | CNXT | 0.3065 | 0.4590 | 0.0521 | 0.2420 | — |
| 18 | e4b_composite | FamB | 0.3062 | 0.4466 | 0.0137 | 0.9461 | — |
| 19 | e3b_centroid | FamB | 0.3049 | 0.4459 | 0.0130 | 0.9457 | — |
| 20 | e4_comp_cnxt | CNXT | 0.3045 | 0.4643 | 0.0172 | 0.9445 | — |
| 21 | e10_pf_cnxt | CNXT | 0.3043 | 0.4593 | 0.0297 | 0.2093 | — |
| 22 | e5_learn_cnxt | CNXT | 0.2998 | 0.4622 | 0.0169 | 0.9432 | — |
| 23 | e3_creg_cnxt | CNXT | 0.2982 | 0.4610 | 0.0173 | 0.9426 | — |

### 3.2 ConvNeXt vs ResNet (10 paired experiments)

| Pair | ResNet cm_R@1 | CNXT cm_R@1 | Δ | img→c Δ | Winner cm_R@1 |
|------|--------------|-------------|---|---------|---------------|
| e1_pairwise | 0.4430 | 0.4783 | **+0.035** | +0.065 | **CNXT** |
| e3_centroid_reg | 0.3146 | 0.2982 | -0.016 | -0.005 | ResNet |
| e4_composite | 0.3167 | 0.3045 | -0.012 | -0.003 | ResNet |
| e5_learnable | 0.3146 | 0.2998 | -0.015 | -0.001 | ResNet |
| e6_fuzzy | 0.3517 | 0.3361 | -0.016 | +0.019 | ResNet |
| e7_lyapunov | 0.3508 | 0.3065 | **-0.044** | +0.109 | ResNet |
| e8c_active | 0.4051 | 0.3574 | -0.048 | **+0.582** | ResNet |
| e9_potential | 0.3309 | 0.3091 | -0.022 | -0.003 | ResNet |
| e10_pot_fuzzy | 0.3412 | 0.3043 | -0.037 | +0.076 | ResNet |
| e8c_low_va | 0.4159 | 0.3601 | -0.056 | **+0.489** | ResNet |

**Вывод:** ConvNeXt побеждает ТОЛЬКО в pairwise E1 (+3.5pp, новый рекорд). Проигрывает в 9/10 по cm_R@1.
Парадокс E8c_cnxt: img→centroid R@1 = 0.94 (ResNet: 0.36), но cm_R@1 хуже. Избыточное визуальное выравнивание сжимает пространство (D_intra=0.017 vs 0.040).

### 3.3 E8c — главный бенефициар EXP-008

| Metric | Sweep10 (20%) | Sweep11 (10%) | Изменение |
|--------|--------------|---------------|-----------|
| E8c_low_va cm_R@1 | 0.376 (rank #7) | **0.416 (rank #3)** | +4.0pp, rank ↑4 |
| E8c_active cm_R@1 | 0.376 (rank #6) | **0.405 (rank #4)** | +2.9pp, rank ↑2 |
| Gap to E1 | 9.5pp | **2.7pp** | Gap ↓71% |

При том что данных вдвое меньше (10% vs 20%). Причина: γ÷5 снял elastic constraint, MLP-консеквенты получили свободу корректировки.

### 3.4 Controller ranking (E6 vs E8c vs E7 vs E10)

| Experiment | cm_R@1 | % of E1 baseline |
|-----------|--------|-------------------|
| E8c_low_va | 0.4159 | 93.9% |
| E8c_active | 0.4051 | 91.4% |
| E6_fuzzy | 0.3517 | 79.4% |
| E7_lyapunov | 0.3508 | 79.2% |
| E10_pot_fuzzy | 0.3412 | 77.0% |

Nonlinear MLP consequents (E8c) >>> Linear consequents (E6) >>> Potential+Fuzzy (E10).

### 3.5 Family A vs B (stable across sweeps)

| Pair | Fam A | Fam B | Δ |
|------|-------|-------|---|
| E1 pairwise | 0.4430 | 0.4224 | +2.1pp |
| E3 centroid | 0.3146 | 0.3049 | +1.0pp |
| E4 composite | 0.3167 | 0.3062 | +1.1pp |

Family A стабильно лучше (+1-2pp). Family B при 46% параметров — достойный trade-off.

## 4. Верификация гипотез H41-H50

| ID | Вердикт | Данные |
|----|---------|--------|
| H41 | **ПОДТВЕРЖДЕНА** | img→lat: CNXT=0.3364 vs ResNet=0.1978 (+70%) |
| H42 | **ПОДТВЕРЖДЕНА** | E1_cnxt=0.4783 > 0.47 (новый рекорд) |
| H43 | **ОТВЕРГНУТА** | E10_cnxt=0.3043 < E10=0.3412 |
| H44 | **ПОДТВЕРЖДЕНА** | E6_cnxt=0.3361 > E3_cnxt=0.2982 (+3.8pp) |
| H45 | **ОТВЕРГНУТА** | ConvNeXt вредит centroid, помогает pairwise |
| H46 | **ЧАСТИЧНО** | Нет прямых данных L_va/L_total в логах |
| H47 | **ЧАСТИЧНО** | E8c jump #7→#3 — косвенное подтверждение |
| H48 | **ПОДТВЕРЖДЕНА** | E6-E4=+0.035 (было +0.025) |
| H49 | **НЕ ПРОВЕРЕНА** | Нужен TensorBoard |
| H50 | **ОТВЕРГНУТА** | E10=0.341<0.42, но E8c=0.416≈0.42 |

**Итого:** 4/10 подтверждены, 3/10 отвергнуты, 2/10 частично, 1/10 не проверена.

## 5. Ключевые находки

### 5.1 ConvNeXt-Pico — специалист pairwise
ConvNeXt-Pico (8.5M, 512-dim, ImageNet-22k pretrained) даёт лучшие low-level визуальные фичи, что критично для pairwise InfoNCE где модальности соревнуются напрямую. Но при добавлении alignment/radial/VA регуляризации ConvNeXt embeddings перетягиваются (over-constrained), и cm_R@1 падает на 1-6pp vs ResNet.

### 5.2 Нелинейные консеквенты (E8c) — прорыв
MLP per rule (φ_r: R^18 → R^11) + ослабленный elastic (γ÷5) = лучшая регуляризованная модель. Gap к pairwise: 2.7pp (было 9.5pp). Линейные A_r·s_t + b_r (E6) по-прежнему ограничены.

### 5.3 Два режима эмбеддингов — фундаментальный компромисс
- **Pairwise** (D_intra>1): модальности "живут" далеко от центроида, cm_R@1 высокий, collapse умеренный
- **Regularized** (D_intra<0.05): модальности прижаты к центроиду, collapse=0, но cm_R@1 ниже

E8c_low_va находится в sweet spot: D_intra=0.04 (30× ниже pairwise, но 2.5× выше E3-E5).

### 5.4 Lyapunov по-прежнему инертен
E7≈E6 (0.3508 vs 0.3517, δ=0.0009). Soft penalty V_t недостаточен для изменения динамики. Нужна другая формулировка или отказ от Lyapunov в пользу nonlinear consequents.

## 6. Следующие шаги

1. **Full training** лучших 3-5 экспериментов (100% данных, 10-15 эпох, 3 seeds)
2. **TensorBoard анализ:** H46 (L_va/L_total), H47 (lambda drift), H49 (weight_norm)
3. **Попробовать E8c + ConvNeXt-Pico в pairwise режиме** (без centroid reg)
4. **Отказ от E7 Lyapunov** — заменить на nonlinear E8c variant
5. **Checkpoint cleanup** — оставить только best_model.pt

---

*exp008_sweep11_10pct_5ep.md | 2026-03-16*
