# EXP-012: Full Training Run — Plan

> Дата: 2026-03-17 (updated 2026-03-18)
> Статус: запуск (attempt 2 — fixes: batched eval, @no_grad, reduced batch)
> Предыдущий: EXP-011 (sweep13, 32/32 ok)

---

## 1. Эксперименты

| # | Experiment | Backbone | Batch | cm@1 (10%) | Роль |
|---|---|---|---|---|---|
| 1 | e1_pairwise_cnxt | ConvNeXt-Pico | 96 | 0.4727 | best overall |
| 2 | e1_pairwise | ResNet18 | 192 | 0.4422 | backbone ablation |
| 3 | e1b_pairwise | ResNet18 | 192 | 0.4213 | encoder architecture (Family B) |
| 4 | e8c_low_va | ResNet18 | 192 | 0.4140 | best centroid + controller |
| 5 | e8c_low_va_cnxt | ConvNeXt-Pico | 96 | 0.3525 | ConvNeXt + centroid at scale |
| 6 | e9_potential | ResNet18 | 192 | 0.3294 | potential functions |

## 2. Параметры

- **Data:** 100% (972,711 objects)
- **Split:** 924,076 train (95%) / 29,181 val (3%) / 19,454 test (2%, held-out)
- **Epochs:** 15
- **Eval:** every 1000 steps (val set), final eval on held-out test
- **Eval method:** batched cosine search (RAG-style, chunk=256), NOT full NxN matrix
- **Checkpoints:** keep_best=3 + final_model.pt (always saved)
- **TensorBoard:** run_tag=full → "full_{experiment}_s42_..."
- **GPU:** RTX 3090 (24GB)

## 2.1. Fixes from attempt 1 (2026-03-18)

- `evaluate()` now uses `@torch.no_grad()` (was building computation graphs)
- `recall_at_k()` uses batched cosine search (chunk=256) instead of full NxN matrix
- `torch.cuda.empty_cache()` after each eval
- ResNet batch: 256 → 192 (OOM at step 271 due to CUDA fragmentation)
- Old matrix method preserved as `recall_at_k_matrix()` for sweep comparison

## 3. Estimated time

- ResNet18 (bs=192): ~65 min/epoch, 15 ep = ~16h
- ConvNeXt (bs=96): ~72 min/epoch, 15 ep = ~18h
- Total: 4×16h + 2×18h = **~100h (~4.2 days)**

## 4. Disk budget

- Checkpoints: 6 exp × 4 ckpts × ~4.5GB = ~108GB
- Free disk: 186GB → ~78GB remaining
- TensorBoard: ~200MB (negligible)

## 5. Key questions this run answers

1. Do 10%-sweep rankings hold at 100% data?
2. Does ConvNeXt + centroid (e8c_low_va_cnxt) improve at scale? (Paradox from sweep13)
3. What is the gap between pairwise and centroid-based at full data?
4. Does e9_potential benefit from more data?
5. Final metrics for paper: R@1, R@10, mod→centroid, cross-modal matrix

---

*EXP-012 plan | SciLibMath_v2 | 2026-03-17*
