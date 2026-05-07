# EXP-013: Full Training Plan

> Date: 2026-04-27
> GPU: RTX 3090 24GB
> Expected duration: ~80h (4 experiments × ~20h sequential)

---

## Experiments

| # | Config | Method | Sweep cm@1 | Notes |
|---|--------|--------|-----------|-------|
| 1 | bl3_uncertainty | Uncertainty Weighting (Kendall 2018) | 0.4121 | 5 learnable log-variance params |
| 2 | bl1_gradnorm | GradNorm (Chen 2018) | 0.4084 | Gradient norm matching, α=1.5 |
| 3 | e8cf_differentiable | E8cf_unc (v3) — reaggregate + softmax(u_t) | 0.5084 | MLP decorative, best sweep metric |
| 4 | e8cf_real | E8cf_real (v5) — residual MLP consequents | pending | MLP = linear + learnable correction |

## Common Overrides (full run)

```
data.dataset_fraction=1.0
data.test_fraction=0.02
data.val_fraction=0.03
data.batch_size=64
data.num_workers=8
training.epochs=15
eval.eval_every_steps=1000
checkpoint.enabled=true
checkpoint.keep_best=3
run_tag=full
```

## Reference Results (previous full runs)

| Method | cm@1 (full) | Epochs |
|--------|-------------|--------|
| E8c_low_va_cnxt | 0.7070 | 15 |
| E1_pairwise_cnxt | 0.6971 | 15 |
| E9_potential | 0.6891 | 15 |
| E8c_low_va (ResNet) | 0.6713 | 15 |
| E1_pairwise (ResNet) | 0.6682 | 15 |

## Key Questions

1. Does BL3/BL1 advantage over E8c on sweep (10%) hold on full data?
2. Does e8cf_unc (0.5084 sweep) maintain its lead on full data?
3. Does e8cf_real (residual MLP) actually learn meaningful corrections?
4. How do weight trajectories differ between sweep and full?

## Run Script

```bash
nohup bash run_full_exp013.sh > logs/exp013_full_run.log 2>&1 &
```

## TensorBoard Naming

Full runs will appear as:
- `full_bl3_uncertainty_s42_<ts>`
- `full_bl1_gradnorm_s42_<ts>`
- `full_e8cf_differentiable_s42_<ts>`
- `full_e8cf_real_s42_<ts>`

(run_tag=full prefixes the experiment name automatically)

## Rollback

```bash
# Kill running full experiment
kill $(ps aux | grep 'run_full_exp013' | grep -v grep | awk '{print $2}')

# Completed runs are in runs/full_* — do NOT delete
# Failed runs: check logs/exp013_full_*.log
```
