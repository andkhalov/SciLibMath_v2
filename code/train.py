"""Main training loop for SciLibMath_v2 experiments E1-E7.
Ref: MATH.md M.3 (experiments), TZ.md Sec 6 (implementation)

Usage:
    python code/train.py --config configs/e4_composite_static.yaml
    python code/train.py --config configs/e1_pairwise.yaml seed=123
"""

import argparse
import sys
import warnings
from pathlib import Path

# Add dataset package to path (scilibrumodal_v2_data lives in data/scilibrumodal-v2/)
_dataset_pkg = Path(__file__).resolve().parent.parent / "data" / "scilibrumodal-v2"
if _dataset_pkg.exists():
    sys.path.insert(0, str(_dataset_pkg))

# Suppress benign scheduler warning (OneCycleLR with GradScaler)
warnings.filterwarnings("ignore", "Detected call of `lr_scheduler.step\\(\\)` before")

from tqdm import tqdm

import torch
import torch.nn.functional as F

from utils import load_config, save_config, set_seed, get_device, get_amp_context
from utils import CheckpointState, save_checkpoint, load_checkpoint, manage_checkpoints
from data import create_dataloaders
from models import FamilyA, MODALITIES
from losses import PairwiseInfoNCE, CentroidInfoNCE, CompositeLoss, LossMixerComposite
from losses.alignment import AlignmentLoss, RadialLoss, AntiCollapseLoss
from metrics import compute_retrieval_metrics, compute_geometry_metrics
from experiment_logging import TBLogger, S3BackupDaemon
from controller import StateTracker, TSFuzzyController, LyapunovRegularizer


def build_loss_fn(cfg):
    """Build loss function based on experiment type."""
    exp = cfg.experiment
    loss_cfg = cfg.loss

    if exp == "e1_pairwise":
        return PairwiseInfoNCE(tau=loss_cfg.tau), "pairwise"

    elif exp == "e2_centroid":
        return CentroidInfoNCE(tau=loss_cfg.tau), "centroid"

    elif exp in ("e3_centroid_reg", "e4_composite_static", "e6_fuzzy", "e7_lyapunov"):
        weights = dict(loss_cfg.get("weights", {}))
        return CompositeLoss(
            tau=loss_cfg.tau,
            lambda_align=loss_cfg.lambda_align,
            lambda_rad=loss_cfg.lambda_rad,
            lambda_reg=loss_cfg.lambda_reg,
            lambda_va=loss_cfg.lambda_va,
            modality_weights=weights,
            w_g=loss_cfg.get("w_g", 1.0),
        ), "composite"

    elif exp == "e5_composite_learnable":
        return CompositeLoss(
            tau=loss_cfg.tau,
            lambda_align=loss_cfg.lambda_align,
            lambda_rad=loss_cfg.lambda_rad,
            lambda_reg=loss_cfg.lambda_reg,
            lambda_va=loss_cfg.lambda_va,
        ), "mixer"

    else:
        raise ValueError(f"Unknown experiment: {exp}")


def train_step_pairwise(model, batch, loss_fn, device):
    """E1: Pairwise InfoNCE step."""
    out = model(batch)
    result = loss_fn(out["embeddings"])
    result["visual_align_loss"] = out["visual_align_loss"]
    return result


def train_step_centroid(model, batch, loss_fn, device):
    """E2: Centroid InfoNCE step."""
    out = model(batch)
    result = loss_fn(out["embeddings"], out["centroid"])
    result["visual_align_loss"] = out["visual_align_loss"]
    return result


def train_step_composite(model, batch, loss_fn, device):
    """E3/E4/E6/E7: Composite loss step."""
    out = model(batch)
    result = loss_fn(out["embeddings"], out["centroid"], out["visual_align_loss"])
    return result


@torch.no_grad()
def evaluate(model, test_loader, device, ks=[1, 3, 10]):
    """Run evaluation on test set: retrieval + geometry metrics."""
    model.eval()
    all_embeddings = {m: [] for m in MODALITIES}
    all_centroids = []

    for batch in test_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = model(batch)

        for m in MODALITIES:
            all_embeddings[m].append(out["embeddings"][m].cpu())
        all_centroids.append(out["centroid"].cpu())

    # Concatenate
    embeddings = {m: torch.cat(all_embeddings[m], dim=0) for m in MODALITIES}
    centroids = torch.cat(all_centroids, dim=0)

    # Compute metrics
    retrieval = compute_retrieval_metrics(embeddings, centroids, ks)
    geometry = compute_geometry_metrics(embeddings, centroids)

    model.train()
    return {**retrieval, **geometry}


def main():
    parser = argparse.ArgumentParser(description="SciLibMath_v2 Training")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    args, overrides = parser.parse_known_args()

    # Load config
    cfg = load_config(args.config, overrides if overrides else None)
    print(f"Experiment: {cfg.experiment}, Seed: {cfg.seed}")

    # Setup
    set_seed(cfg.seed, cfg.get("deterministic", False))
    device = get_device(cfg.device)
    print(f"Device: {device}")

    # Data
    print("Loading data...")
    train_loader, test_loader, dataset_size = create_dataloaders(
        data_dir=cfg.data.dataset_path,
        image_root=cfg.data.get("image_root"),
        batch_size=cfg.data.batch_size,
        test_fraction=cfg.data.test_fraction,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.get("pin_memory", True),
        seed=cfg.seed,
        tokenizer_name=cfg.model.get("text_backbone", "sentence-transformers/all-MiniLM-L6-v2"),
        max_length=cfg.data.get("max_length", 256),
    )
    print(f"Dataset: {dataset_size} total, train: {len(train_loader.dataset)}, test: {len(test_loader.dataset)}")

    # Model
    print("Building model...")
    model = FamilyA(
        text_backbone=cfg.model.text_backbone,
        visual_backbone=cfg.model.visual_backbone,
        visual_pretrained=cfg.model.visual_pretrained,
        embedding_dim=cfg.model.embedding_dim,
        projection_hidden_dim=cfg.model.projection_hidden_dim,
        text_unfreeze_ratio=cfg.model.get("text_unfreeze_ratio", 1.0),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Loss
    loss_fn, loss_type = build_loss_fn(cfg)
    print(f"Loss type: {loss_type}")

    # E5: LossMixer wrapper
    mixer = None
    if loss_type == "mixer":
        mixer = LossMixerComposite(
            tau=cfg.loss.tau,
            lambda_va=cfg.loss.lambda_va,
        ).to(device)

    # E6-E7: Fuzzy controller
    controller = None
    state_tracker = None
    lyapunov = None
    if cfg.experiment in ("e6_fuzzy", "e7_lyapunov"):
        controller = TSFuzzyController(
            alpha=cfg.get("controller", {}).get("alpha", 0.01),
            device=device,
        )
        state_tracker = StateTracker(beta=0.99, device=device)

    if cfg.experiment == "e7_lyapunov":
        lyapunov = LyapunovRegularizer(
            penalty_weight=cfg.get("lyapunov", {}).get("penalty_weight", 0.1),
            device=device,
        )

    # Optimizer
    param_groups = model.get_param_groups(
        lr=cfg.training.lr,
        lr_embed_ratio=cfg.model.get("lr_embed_ratio", 0.1),
    )
    if mixer:
        param_groups.append({"params": mixer.parameters(), "lr": cfg.training.lr})

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=cfg.training.weight_decay,
    )

    # Scheduler
    total_steps = len(train_loader) * cfg.training.epochs
    warmup_steps = cfg.training.get("warmup_steps", 500)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[pg["lr"] for pg in param_groups],
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps if total_steps > 0 else 0.1,
    )

    # Mixed precision
    amp_ctx, scaler = get_amp_context(cfg.get("mixed_precision", True), device)

    # Logging
    logger = TBLogger(cfg)
    save_config(cfg, logger.log_dir / "config.yaml")
    print(f"TensorBoard: {logger.log_dir}")

    # S3 backup
    s3_daemon = None
    if cfg.logging.get("s3_backup", False):
        s3_daemon = S3BackupDaemon(
            local_dir=cfg.logging.tensorboard_dir,
            remote=cfg.logging.get("s3_remote", "scilib-store"),
            bucket=cfg.logging.get("s3_bucket", "scilibmath-v2-logs"),
            interval_minutes=cfg.logging.get("backup_every_minutes", 30),
        )
        s3_daemon.start()
        print("S3 backup daemon started")

    # Checkpoint state
    ckpt_state = CheckpointState()
    if cfg.checkpoint.get("resume_from"):
        print(f"Resuming from {cfg.checkpoint.resume_from}")
        ckpt_state = load_checkpoint(
            cfg.checkpoint.resume_from, model, optimizer, scheduler, scaler, device
        )

    # Select train step function
    if loss_type == "pairwise":
        step_fn = train_step_pairwise
    elif loss_type == "centroid":
        step_fn = train_step_centroid
    else:
        step_fn = train_step_composite

    # ==================== Training Loop ====================
    print(f"\nStarting training: {cfg.training.epochs} epochs, {len(train_loader)} steps/epoch")
    eval_every = cfg.eval.get("eval_every_steps", 200)
    prev_lambda = None

    for epoch in range(ckpt_state.epoch, cfg.training.epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.epochs}")
        for step_in_epoch, batch in enumerate(pbar):
            global_step = ckpt_state.global_step

            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Forward + loss
            optimizer.zero_grad()
            with amp_ctx:
                loss_dict = step_fn(model, batch, loss_fn, device)

                # E1/E2: add visual align loss manually
                if loss_type in ("pairwise", "centroid"):
                    total_loss = loss_dict["loss"] + cfg.loss.get("lambda_va", 0.1) * loss_dict["visual_align_loss"]
                    loss_dict["loss"] = total_loss
                    loss_dict["loss_va"] = loss_dict["visual_align_loss"]

                # E5: LossMixer reweighting
                if mixer and loss_type == "mixer":
                    loss_dict = mixer(loss_dict, loss_dict.get("loss_va", torch.tensor(0.0, device=device)))

                # E6-E7: Fuzzy controller
                if controller and state_tracker:
                    s_t = state_tracker.update(loss_dict)
                    curr_lambda = loss_fn.get_lambda_vector().to(device)

                    new_lambda, u_t, h_bar = controller.step(s_t, curr_lambda)
                    loss_fn.set_lambda_vector(new_lambda)

                    # Log controller state
                    if global_step % 10 == 0:
                        for i, name in enumerate(["τ", "λ_align", "λ_rad", "λ_reg", "λ_va",
                                                   "w_en", "w_ru", "w_lean", "w_latex", "w_img", "w_g"]):
                            logger.log_scalar(f"controller/lambda/{name}", new_lambda[i].item(), global_step)
                        for r in range(7):
                            logger.log_scalar(f"controller/rule_activation/R{r}", h_bar[r].item(), global_step)

                    # E7: Lyapunov penalty
                    if lyapunov:
                        delta_lam = (new_lambda - curr_lambda) if prev_lambda is not None else torch.zeros_like(new_lambda)
                        mod_weights = new_lambda[5:10]  # w_en..w_img
                        L_norm = loss_dict["loss"].detach().item()
                        penalty, lyap_info = lyapunov.get_penalty(L_norm, delta_lam, mod_weights)
                        loss_dict["loss"] = loss_dict["loss"] + penalty
                        logger.log_scalar("lyapunov/V_t", lyap_info["V_t"], global_step)
                        logger.log_scalar("lyapunov/delta_V", lyap_info["delta_V"], global_step)

                    prev_lambda = curr_lambda

                loss = loss_dict["loss"]

            # Backward
            if scaler:
                scaler.scale(loss).backward()
                if cfg.training.get("gradient_clip", 0) > 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.training.gradient_clip
                    )
                else:
                    grad_norm = torch.tensor(0.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.training.get("gradient_clip", 0) > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.training.gradient_clip
                    )
                else:
                    grad_norm = torch.tensor(0.0)
                optimizer.step()

            scheduler.step()

            # Logging
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if global_step % 10 == 0:
                logger.log_loss_components(loss_dict, global_step)
                logger.log_training_stats({
                    "lr": optimizer.param_groups[-1]["lr"],
                    "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else grad_norm,
                    "epoch": epoch,
                }, global_step)

            # Evaluation
            if global_step > 0 and global_step % eval_every == 0:
                print(f"\n[Step {global_step}] Evaluating...")
                metrics = evaluate(model, test_loader, device, cfg.eval.retrieval_k)
                logger.log_eval_metrics(metrics, global_step)

                # Check for best model
                key_metric = metrics.get(f"centroid_R@{cfg.eval.retrieval_k[0]}", 0)
                if key_metric > ckpt_state.best_metric:
                    ckpt_state.best_metric = key_metric
                    ckpt_state.best_epoch = epoch
                    save_checkpoint(
                        Path(cfg.checkpoint.dir) / cfg.experiment / "best_model.pt",
                        model, optimizer, scheduler, ckpt_state, cfg, scaler,
                    )
                    print(f"  New best: R@{cfg.eval.retrieval_k[0]}={key_metric:.4f}")

                # Print key metrics
                for k in cfg.eval.retrieval_k:
                    r_key = f"centroid_R@{k}"
                    if r_key in metrics:
                        print(f"  {r_key}: {metrics[r_key]:.4f}", end="  ")
                print(f"  D_intra: {metrics.get('D_intra', 0):.4f}  collapse: {metrics.get('collapse_score', 0):.4f}")

            ckpt_state.global_step += 1

        # End of epoch
        avg_loss = epoch_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch+1} done. Avg loss: {avg_loss:.4f}")

        # Save epoch checkpoint
        ckpt_state.epoch = epoch + 1
        save_checkpoint(
            Path(cfg.checkpoint.dir) / cfg.experiment / f"epoch_{epoch+1:03d}.pt",
            model, optimizer, scheduler, ckpt_state, cfg, scaler,
        )
        manage_checkpoints(
            Path(cfg.checkpoint.dir) / cfg.experiment,
            keep_best=cfg.checkpoint.get("keep_best", 3),
        )

        logger.flush()

    # Final evaluation
    print("\n=== Final Evaluation ===")
    final_metrics = evaluate(model, test_loader, device, cfg.eval.retrieval_k)
    logger.log_eval_metrics(final_metrics, ckpt_state.global_step)

    for name, value in sorted(final_metrics.items()):
        print(f"  {name}: {value:.4f}")

    # Cleanup
    logger.close()
    if s3_daemon:
        s3_daemon.stop()
        print("S3 backup: final sync done")

    print(f"\nDone! Best R@{cfg.eval.retrieval_k[0]}: {ckpt_state.best_metric:.4f} (epoch {ckpt_state.best_epoch+1})")
    print(f"Logs: {logger.log_dir}")


if __name__ == "__main__":
    main()
