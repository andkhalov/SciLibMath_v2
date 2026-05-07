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
from data import create_dataloaders, prepare_tokenizers, fvt_initialize
from models import FamilyA, FamilyB, MODALITIES
from losses import PairwiseInfoNCE, CentroidInfoNCE, CompositeLoss, LossMixerComposite
from losses.alignment import AlignmentLoss, RadialLoss, AntiCollapseLoss
from metrics import compute_retrieval_metrics, compute_geometry_metrics
from experiment_logging import TBLogger, S3BackupDaemon
from controller import StateTracker, TSFuzzyController, LyapunovRegularizer
from baselines import GradNormBalancer, PCGradOptimizer, UncertaintyWeighting


def build_loss_fn(cfg):
    """Build loss function based on experiment type."""
    exp = cfg.experiment
    loss_cfg = cfg.loss

    if exp in ("e1_pairwise", "e1b_pairwise", "e1_pairwise_cnxt"):
        return PairwiseInfoNCE(tau=loss_cfg.tau), "pairwise"

    elif exp in ("e2_centroid", "e2b_centroid"):
        return CentroidInfoNCE(
            tau=loss_cfg.tau,
            p_drop=loss_cfg.get("p_drop", 0.3),
        ), "centroid"

    elif exp in ("e3_centroid_reg", "e3b_centroid_reg",
                 "e4_composite_static", "e4b_composite_static",
                 "e6_fuzzy", "e7_lyapunov", "e8_nonlinear",
                 "e10_potential_fuzzy",
                 "e3c_low_va", "e6c_low_va", "e8c_active",
                 "e10c_low_va",
                 "e6_rho03", "e8c_rho03", "e8c_low_va",
                 "e8c_rho03_low_va", "e8c_boost",
                 "e8c_pairwise", "e6_low_elastic",
                 # ConvNeXt backbone variants
                 "e3_centroid_reg_cnxt", "e4_composite_static_cnxt",
                 "e6_fuzzy_cnxt", "e7_lyapunov_cnxt", "e8c_active_cnxt",
                 "e10_potential_fuzzy_cnxt", "e8c_low_va_cnxt",
                 # H-hypothesis experiments (sweep13)
                 "h51_wmin05", "h51_wmin06",
                 "h52_low_align", "h52_low_align_e4",
                 "h53_combined", "h53_combined_06",
                 "h55_img_warmup",
                 # BL baselines (EXP-013)
                 "bl1_gradnorm", "bl2_pcgrad", "bl3_uncertainty",
                 # E8cf: differentiable controller (EXP-013)
                 "e8cf_differentiable", "e8cf_v6", "e8cf_v7", "e8cf_v8",
                 "e8cf_v7d", "e8cf_v8d"):
        weights = dict(loss_cfg.get("weights", {}))
        use_potential = loss_cfg.get("use_potential", False)
        return CompositeLoss(
            tau=loss_cfg.tau,
            lambda_align=loss_cfg.get("lambda_align", 0.3),
            lambda_rad=loss_cfg.get("lambda_rad", 0.1),
            lambda_reg=loss_cfg.lambda_reg,
            lambda_va=loss_cfg.lambda_va,
            modality_weights=weights,
            w_g=loss_cfg.get("w_g", 1.0),
            p_drop=loss_cfg.get("p_drop", 0.3),
            C_clip=loss_cfg.get("C_clip", 10.0),
            rho=loss_cfg.get("rho", 0.1),
            alpha_tau=loss_cfg.get("alpha_tau", 0.5),
            tau_target=loss_cfg.get("tau_target", 0.0),
            tau_min=loss_cfg.get("tau_min", 0.01),
            tau_max=loss_cfg.get("tau_max", 0.5),
            use_potential=use_potential,
            k_a=loss_cfg.get("k_a", 1.0),
            k_r=loss_cfg.get("k_r", 0.1),
            contrast_weight=loss_cfg.get("contrast_weight", 1.0),
            contrast_mode=loss_cfg.get("contrast_mode", "centroid"),
            align_warmup_steps=loss_cfg.get("align_warmup_steps", 0),
        ), "composite"

    elif exp in ("e5_composite_learnable", "e9_potential",
                 "e5_composite_learnable_cnxt", "e9_potential_cnxt"):
        use_potential = loss_cfg.get("use_potential", False)
        return CompositeLoss(
            tau=loss_cfg.tau,
            lambda_align=loss_cfg.get("lambda_align", 0.3),
            lambda_rad=loss_cfg.get("lambda_rad", 0.1),
            lambda_reg=loss_cfg.lambda_reg,
            lambda_va=loss_cfg.lambda_va,
            p_drop=loss_cfg.get("p_drop", 0.3),
            C_clip=loss_cfg.get("C_clip", 10.0),
            rho=loss_cfg.get("rho", 0.1),
            alpha_tau=loss_cfg.get("alpha_tau", 0.5),
            tau_target=loss_cfg.get("tau_target", 0.0),
            tau_min=loss_cfg.get("tau_min", 0.01),
            tau_max=loss_cfg.get("tau_max", 0.5),
            use_potential=use_potential,
            k_a=loss_cfg.get("k_a", 1.0),
            k_r=loss_cfg.get("k_r", 0.1),
            contrast_weight=loss_cfg.get("contrast_weight", 1.0),
        ), "mixer"

    else:
        raise ValueError(f"Unknown experiment: {exp}")


def train_step_pairwise(model, batch, loss_fn, device):
    """E1: Pairwise InfoNCE step. _infonce_loss normalizes internally."""
    out = model(batch)
    result = loss_fn(out["embeddings"])  # embeddings unnormalized; loss normalizes
    result["visual_align_loss"] = out["visual_align_loss"]
    return result


def train_step_centroid(model, batch, loss_fn, device):
    """E2: Centroid InfoNCE step. Centroid unnormalized; loss normalizes internally."""
    out = model(batch)
    result = loss_fn(out["embeddings"], out["centroid"])
    result["visual_align_loss"] = out["visual_align_loss"]
    return result


def train_step_composite(model, batch, loss_fn, device):
    """E3/E4/E6/E7: Composite loss step. Gets UNNORMALIZED embeddings+centroid per M.0.4."""
    out = model(batch)
    result = loss_fn(out["embeddings"], out["centroid"], out["visual_align_loss"])
    return result


@torch.no_grad()
@torch.no_grad()
def evaluate(model, test_loader, device, ks=[1, 3, 10]):
    """Run evaluation on test set: retrieval + geometry metrics.

    Per MATH.md M.0.4: model returns unnormalized embeddings/centroids.
    - Retrieval: uses normalized (cosine similarity via batched search)
    - Geometry: uses unnormalized (D_intra, D_inter in euclidean)

    Returns (metrics_dict, embeddings_raw, centroids_raw) for visualization.
    """
    model.eval()
    all_embeddings = {m: [] for m in MODALITIES}
    all_centroids = []

    for batch in test_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = model(batch)

        for m in MODALITIES:
            all_embeddings[m].append(out["embeddings"][m].cpu())
        all_centroids.append(out["centroid"].cpu())

    # Concatenate (unnormalized, on CPU)
    embeddings = {m: torch.cat(all_embeddings[m], dim=0) for m in MODALITIES}
    centroids = torch.cat(all_centroids, dim=0)

    # Geometry uses unnormalized embeddings/centroids
    geometry = compute_geometry_metrics(embeddings, centroids)

    # Retrieval uses normalized (cosine similarity via batched search)
    embeddings_norm = {m: F.normalize(emb, dim=-1) for m, emb in embeddings.items()}
    centroids_norm = F.normalize(centroids, dim=-1)
    retrieval = compute_retrieval_metrics(embeddings_norm, centroids_norm, ks)

    model.train()
    return {**retrieval, **geometry}, embeddings, centroids


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

    # Tokenizers (MATH.md M.2.3)
    family = cfg.model.get("family", "A")
    text_backbone = cfg.model.get("text_backbone", "mlsa-iai-msu-lab/sci-rus-tiny3.5-zh")
    tok_cfg = cfg.get("tokenizer", {})
    lean_vocab_size = tok_cfg.get("lean_vocab_size", None)
    latex_vocab_size = tok_cfg.get("latex_vocab_size", None)

    tokenizers_dict = None
    if lean_vocab_size or latex_vocab_size:
        print("Preparing custom tokenizers (MATH.md M.2.3)...")
        tokenizers_dict = prepare_tokenizers(
            data_dir=cfg.data.dataset_path,
            base_name=text_backbone,
            lean_vocab_size=lean_vocab_size or 16000,
            latex_vocab_size=latex_vocab_size or 16000,
            cache_dir=tok_cfg.get("cache_dir", "tokenizers"),
            max_corpus_samples=tok_cfg.get("max_corpus_samples", 0),
        )
        print(f"  Base vocab: {len(tokenizers_dict['base'])}, "
              f"Lean vocab: {len(tokenizers_dict['lean'])} (+{len(tokenizers_dict['lean_new_tokens'])}), "
              f"LaTeX vocab: {len(tokenizers_dict['latex'])} (+{len(tokenizers_dict['latex_new_tokens'])})")

    # Data
    print("Loading data...")
    data_result = create_dataloaders(
        data_dir=cfg.data.dataset_path,
        image_root=cfg.data.get("image_root"),
        batch_size=cfg.data.batch_size,
        dataset_fraction=cfg.data.get("dataset_fraction", 1.0),
        test_fraction=cfg.data.test_fraction,
        val_fraction=cfg.data.get("val_fraction", 0.0),
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.get("pin_memory", True),
        seed=cfg.seed,
        tokenizer_name=text_backbone,
        max_length=cfg.data.get("max_length", 128),
        tokenizers=tokenizers_dict,
    )
    train_loader, eval_loader, held_out_test_loader, dataset_size = data_result
    # eval_loader = val (if val_fraction>0) or test (backward compat)
    # held_out_test_loader = test (if val_fraction>0) or None
    print(f"Dataset: {dataset_size} total, train: {len(train_loader.dataset)}, eval: {len(eval_loader.dataset)}"
          + (f", held-out test: {len(held_out_test_loader.dataset)}" if held_out_test_loader else ""))

    # Model — dispatch Family A vs Family B
    va_margin = cfg.model.get("visual_align_margin", 1.0)
    print(f"Building model (Family {family})...")
    if family == "B":
        # Family B: shared vocab = union of base + lean_new + latex_new
        shared_vocab_size = None
        if tokenizers_dict:
            shared_vocab_size = max(
                len(tokenizers_dict["lean"]),
                len(tokenizers_dict["latex"]),
            )
        model = FamilyB(
            text_backbone=text_backbone,
            visual_backbone=cfg.model.visual_backbone,
            visual_pretrained=cfg.model.get("visual_pretrained", True),
            embedding_dim=cfg.model.embedding_dim,
            visual_patch_size=cfg.model.get("visual_patch_size", 64),
            visual_patch_stride=cfg.model.get("visual_patch_stride", 32),
            visual_align_margin=va_margin,
            visual_align_targets=list(cfg.model.get("visual_align_targets", ["latex"])),
            align_hidden_dim=cfg.model.get("align_hidden_dim", 512),
            align_dropout=cfg.model.get("align_dropout", 0.1),
            freeze_resnet_layers=cfg.model.get("freeze_resnet_layers", 2),
            shared_vocab_size=shared_vocab_size,
        ).to(device)
        # FVT init for shared encoder (all new tokens)
        if tokenizers_dict:
            all_new = sorted(set(tokenizers_dict["lean_new_tokens"]) | set(tokenizers_dict["latex_new_tokens"]))
            fvt_initialize(model.text_encoder, tokenizers_dict["base"], tokenizers_dict["lean"], all_new)
            print(f"  FVT initialized {len(all_new)} new tokens in shared encoder")
    else:
        # Family A: separate encoders, vocab extension per modality
        _lean_vs = len(tokenizers_dict["lean"]) if tokenizers_dict else None
        _latex_vs = len(tokenizers_dict["latex"]) if tokenizers_dict else None
        model = FamilyA(
            text_backbone=text_backbone,
            visual_backbone=cfg.model.visual_backbone,
            visual_pretrained=cfg.model.get("visual_pretrained", True),
            embedding_dim=cfg.model.embedding_dim,
            visual_patch_size=cfg.model.get("visual_patch_size", 64),
            visual_patch_stride=cfg.model.get("visual_patch_stride", 32),
            visual_align_margin=va_margin,
            visual_align_targets=list(cfg.model.get("visual_align_targets", ["latex"])),
            align_hidden_dim=cfg.model.get("align_hidden_dim", 512),
            align_dropout=cfg.model.get("align_dropout", 0.1),
            freeze_resnet_layers=cfg.model.get("freeze_resnet_layers", 2),
            lean_vocab_size=_lean_vs,
            latex_vocab_size=_latex_vs,
        ).to(device)
        # FVT init per modality
        if tokenizers_dict:
            fvt_initialize(model.text_encoders["lean"], tokenizers_dict["base"],
                           tokenizers_dict["lean"], tokenizers_dict["lean_new_tokens"])
            fvt_initialize(model.text_encoders["latex"], tokenizers_dict["base"],
                           tokenizers_dict["latex"], tokenizers_dict["latex_new_tokens"])
            print(f"  FVT initialized: lean +{len(tokenizers_dict['lean_new_tokens'])}, "
                  f"latex +{len(tokenizers_dict['latex_new_tokens'])}")

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
    if cfg.experiment in ("e6_fuzzy", "e7_lyapunov", "e8_nonlinear", "e10_potential_fuzzy",
                          "e6c_low_va", "e8c_active", "e10c_low_va",
                          "e6_rho03", "e8c_rho03", "e8c_low_va",
                          "e8c_rho03_low_va", "e8c_boost",
                          "e8c_pairwise", "e6_low_elastic",
                          # ConvNeXt backbone with controller
                          "e6_fuzzy_cnxt", "e7_lyapunov_cnxt", "e8c_active_cnxt",
                          "e10_potential_fuzzy_cnxt", "e8c_low_va_cnxt",
                          # H-hypothesis (sweep13): controller-based
                          "h51_wmin05", "h51_wmin06",
                          "h53_combined", "h53_combined_06",
                          "h55_img_warmup",
                          # E8cf: differentiable controller
                          "e8cf_differentiable", "e8cf_v6", "e8cf_v7", "e8cf_v8",
                          "e8cf_v7d", "e8cf_v8d"):
        ctrl_cfg = cfg.get("controller", {})
        total_training_steps = len(train_loader) * cfg.training.epochs
        controller = TSFuzzyController(
            alpha=ctrl_cfg.get("alpha", 0.001),
            device=device,
            warmup_steps=ctrl_cfg.get("warmup_steps", 200),
            step_frequency=ctrl_cfg.get("step_frequency", 10),
            noise_sigma=ctrl_cfg.get("noise_sigma", 0.01),
            noise_anneal=ctrl_cfg.get("noise_anneal", True),
            elastic_gamma=ctrl_cfg.get("elastic_gamma", 0.01),
            total_steps=total_training_steps,
            nonlinear_consequents=ctrl_cfg.get("nonlinear_consequents", False),
            consequent_hidden=ctrl_cfg.get("consequent_hidden", 32),
            init_scale=ctrl_cfg.get("init_scale", 0.01),
            w_min=ctrl_cfg.get("w_min", None),
        )
        state_tracker = StateTracker(beta=0.99, device=device)
        # v7d/v8d: disable bounds for free MLP
        if cfg.get("controller", {}).get("skip_bounds", False):
            controller.skip_bounds = True
            print("Controller: bounds DISABLED (skip_bounds=true)")

    if cfg.experiment in ("e7_lyapunov", "e7_lyapunov_cnxt"):
        lyap_cfg = cfg.get("lyapunov", {})
        lyapunov = LyapunovRegularizer(
            alpha=lyap_cfg.get("alpha", 1.0),
            beta=lyap_cfg.get("beta", 0.1),
            gamma=lyap_cfg.get("gamma", 0.5),
            penalty_weight=lyap_cfg.get("penalty_weight", 0.1),
            xi=lyap_cfg.get("xi", 0.01),
            device=device,
        )

    # BL baselines (EXP-013)
    bl_gradnorm = None
    bl_uncertainty = None
    bl_pcgrad = None
    bl_cfg = cfg.get("baseline", {})
    bl_type = bl_cfg.get("type", None) if bl_cfg else None

    if bl_type == "gradnorm":
        bl_gradnorm = GradNormBalancer(
            alpha=bl_cfg.get("alpha", 1.5),
            lr_w=bl_cfg.get("lr_w", 0.025),
        ).to(device)
        print(f"Baseline: GradNorm (α={bl_cfg.get('alpha', 1.5)}, lr_w={bl_cfg.get('lr_w', 0.025)})")
    elif bl_type == "uncertainty":
        bl_uncertainty = UncertaintyWeighting(
            init_val=bl_cfg.get("init_val", 0.0),
        ).to(device)
        print(f"Baseline: Uncertainty Weighting (init={bl_cfg.get('init_val', 0.0)})")
    elif bl_type == "pcgrad":
        print("Baseline: PCGrad (gradient surgery)")

    # Optimizer
    param_groups = model.get_param_groups(
        lr=cfg.training.lr,
        lr_embed_ratio=cfg.model.get("lr_embed_ratio", 0.1),
        lr_visual_ratio=cfg.model.get("lr_visual_ratio", 0.5),
    )
    if mixer:
        param_groups.append({"params": mixer.parameters(), "lr": cfg.training.lr})
    # E8: add nonlinear consequent parameters to optimizer
    if controller and controller.nl_consequents is not None:
        mlp_lr_mult = cfg.get("controller", {}).get("mlp_lr_multiplier", 1.0)
        param_groups.append({
            "params": controller.nl_consequents.parameters(),
            "lr": cfg.training.lr * mlp_lr_mult,
        })
    # BL3: Uncertainty Weighting log-variance parameters
    if bl_uncertainty:
        param_groups.append({
            "params": bl_uncertainty.parameters(),
            "lr": cfg.training.lr,
        })

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=cfg.training.weight_decay,
    )

    # BL2: PCGrad wrapper (initialized after optimizer)
    if bl_type == "pcgrad":
        bl_pcgrad = PCGradOptimizer(optimizer)
        print("PCGrad optimizer wrapper initialized")

    # Scheduler
    total_steps = len(train_loader) * cfg.training.epochs
    warmup_steps = cfg.training.get("warmup_steps", 500)

    pct_start = min(warmup_steps / total_steps, 0.3) if total_steps > 0 else 0.1
    final_div = cfg.training.get("final_div_factor", 10)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[pg["lr"] for pg in param_groups],
        total_steps=total_steps,
        pct_start=pct_start,
        final_div_factor=final_div,
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

    # L_va curriculum scheduling (MATH.md M.2.4, EXP-008)
    va_warmup_steps = cfg.training.get("va_warmup_steps", 0)
    va_ramp_steps = cfg.training.get("va_ramp_steps", 0)

    # ==================== Training Loop ====================
    print(f"\nStarting training: {cfg.training.epochs} epochs, {len(train_loader)} steps/epoch")
    if va_warmup_steps > 0:
        print(f"L_va curriculum: warmup={va_warmup_steps}, ramp={va_ramp_steps}")
    eval_every = cfg.eval.get("eval_every_steps", 200)
    prev_lambda = None

    # Baseline eval at step 0 (before any training)
    if ckpt_state.epoch == 0 and ckpt_state.global_step == 0:
        print("[Step 0] Baseline evaluation...")
        baseline_metrics, baseline_embs, baseline_cents = evaluate(model, eval_loader, device, cfg.eval.retrieval_k)
        logger.log_eval_metrics(baseline_metrics, 0)
        print(f"  Baseline cm_R@1: {baseline_metrics.get('mean_crossmodal_R@1', 0):.4f}")

    for epoch in range(ckpt_state.epoch, cfg.training.epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.epochs}")
        for step_in_epoch, batch in enumerate(pbar):
            global_step = ckpt_state.global_step

            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # L_va curriculum scheduling (MATH.md M.2.4)
            if va_warmup_steps > 0 and hasattr(loss_fn, "set_va_scale"):
                if global_step < va_warmup_steps:
                    va_scale = 0.0
                elif va_ramp_steps > 0 and global_step < va_warmup_steps + va_ramp_steps:
                    va_scale = (global_step - va_warmup_steps) / va_ramp_steps
                else:
                    va_scale = 1.0
                loss_fn.set_va_scale(va_scale)

            # H55: pass step to loss for alignment warmup
            if hasattr(loss_fn, "set_step"):
                loss_fn.set_step(global_step)

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

                    # E8cf: differentiable reaggregation — MLP consequents get gradient
                    if cfg.experiment == "e8cf_differentiable":
                        e8cf_mu = cfg.get("controller", {}).get("reagg_mu", 1.0)
                        e8cf_min_norm = cfg.get("controller", {}).get("mlp_min_norm", 5.0)
                        e8cf_norm_gamma = cfg.get("controller", {}).get("mlp_norm_gamma", 1.0)
                        # Reaggregate: detached components × differentiable λ
                        # Pass u_t for softmax (unclamped → no dead gradient from bounds)
                        loss_dict["loss"] = loss_fn.reaggregate_with_lambda(
                            loss_dict, new_lambda, u_t=u_t, mu=e8cf_mu,
                        )
                        # Anti-collapse: penalize only if MLP norm drops too low
                        loss_dict["loss"] = loss_dict["loss"] + CompositeLoss.mlp_norm_reg(
                            controller, min_norm=e8cf_min_norm, gamma=e8cf_norm_gamma,
                        )
                        # Update scalars for next step's forward pass
                        loss_fn.set_lambda_vector(new_lambda.detach())
                        # Log diagnostics
                        if global_step % 10 == 0:
                            w_sm = torch.softmax(u_t[5:10].detach(), dim=0) * 5
                            for i, mod in enumerate(MODALITIES):
                                logger.log_scalar(f"e8cf/w_softmax_{mod}", w_sm[i].item(), global_step)
                            mlp_norm = sum(p.norm().item() for mlp in controller.nl_consequents for p in mlp.parameters())
                            logger.log_scalar("e8cf/mlp_weight_norm", mlp_norm, global_step)

                    # E8cf_v6: separate backward — model loss unchanged, MLP trained via reaggregate
                    elif cfg.experiment == "e8cf_v6":
                        e8cf_mu = cfg.get("controller", {}).get("reagg_mu", 10.0)
                        e8cf_min_norm = cfg.get("controller", {}).get("mlp_min_norm", 5.0)
                        e8cf_norm_gamma = cfg.get("controller", {}).get("mlp_norm_gamma", 1.0)
                        # Compute reaggregate loss for MLP (separate from model loss)
                        L_reagg = loss_fn.reaggregate_with_lambda(
                            loss_dict, new_lambda, u_t=u_t, mu=e8cf_mu,
                            model_loss=False,  # return only L_reagg, not loss+L_reagg
                        )
                        L_reagg = L_reagg + CompositeLoss.mlp_norm_reg(
                            controller, min_norm=e8cf_min_norm, gamma=e8cf_norm_gamma,
                        )
                        # Separate backward for MLP (before main backward)
                        L_reagg.backward(retain_graph=True)
                        loss_dict["_mlp_backward_done"] = True
                        # Model loss stays original — no distortion
                        loss_fn.set_lambda_vector(new_lambda.detach())
                        # Log diagnostics
                        if global_step % 10 == 0:
                            w_sm = torch.softmax(u_t[5:10].detach(), dim=0) * 5
                            for i, mod in enumerate(MODALITIES):
                                logger.log_scalar(f"e8cf/w_softmax_{mod}", w_sm[i].item(), global_step)
                            mlp_norm = sum(p.norm().item() for mlp in controller.nl_consequents for p in mlp.parameters())
                            logger.log_scalar("e8cf/mlp_weight_norm", mlp_norm, global_step)
                            logger.log_scalar("e8cf/L_reagg", L_reagg.detach().item(), global_step)

                    # E8cf_v7: per-modality derivative signal — separate backward
                    elif cfg.experiment in ("e8cf_v7", "e8cf_v7d"):
                        L_mlp = loss_fn.compute_derivative_mlp_loss(loss_dict, u_t, s_t)
                        L_mlp = L_mlp + CompositeLoss.mlp_norm_reg(
                            controller, min_norm=cfg.get("controller", {}).get("mlp_min_norm", 5.0),
                            gamma=cfg.get("controller", {}).get("mlp_norm_gamma", 1.0),
                        )
                        L_mlp.backward(retain_graph=True)
                        loss_dict["_mlp_backward_done"] = True
                        loss_fn.set_lambda_vector(new_lambda.detach())
                        if global_step % 10 == 0:
                            logger.log_scalar("e8cf/L_mlp", L_mlp.detach().item(), global_step)
                            for i, mod in enumerate(MODALITIES):
                                logger.log_scalar(f"e8cf/u_t_w_{mod}", u_t[5+i].detach().item(), global_step)
                            mlp_norm = sum(p.norm().item() for mlp in controller.nl_consequents for p in mlp.parameters())
                            logger.log_scalar("e8cf/mlp_weight_norm", mlp_norm, global_step)

                    # E8cf_v8: per-rule derivative signal — separate backward
                    elif cfg.experiment in ("e8cf_v8", "e8cf_v8d"):
                        L_mlp = loss_fn.compute_derivative_per_rule_mlp_loss(
                            loss_dict, u_t, h_bar, controller, s_t,
                        )
                        L_mlp = L_mlp + CompositeLoss.mlp_norm_reg(
                            controller, min_norm=cfg.get("controller", {}).get("mlp_min_norm", 5.0),
                            gamma=cfg.get("controller", {}).get("mlp_norm_gamma", 1.0),
                        )
                        L_mlp.backward(retain_graph=True)
                        loss_dict["_mlp_backward_done"] = True
                        loss_fn.set_lambda_vector(new_lambda.detach())
                        if global_step % 10 == 0:
                            logger.log_scalar("e8cf/L_mlp", L_mlp.detach().item(), global_step)
                            for i, mod in enumerate(MODALITIES):
                                logger.log_scalar(f"e8cf/u_t_w_{mod}", u_t[5+i].detach().item(), global_step)
                            mlp_norm = sum(p.norm().item() for mlp in controller.nl_consequents for p in mlp.parameters())
                            logger.log_scalar("e8cf/mlp_weight_norm", mlp_norm, global_step)

                    else:
                        loss_fn.set_lambda_vector(new_lambda)

                    # Log controller state
                    if global_step % 10 == 0:
                        lam_log = new_lambda.detach() if new_lambda.requires_grad else new_lambda
                        for i, name in enumerate(["τ", "λ_align", "λ_rad", "λ_reg", "λ_va",
                                                   "w_en", "w_ru", "w_lean", "w_latex", "w_img", "w_g"]):
                            logger.log_scalar(f"controller/lambda/{name}", lam_log[i].item(), global_step)
                        for r in range(7):
                            logger.log_scalar(f"controller/rule_activation/R{r}", h_bar[r].item(), global_step)

                    # E7: Lyapunov penalty
                    if lyapunov:
                        delta_lam = (new_lambda.detach() - curr_lambda) if prev_lambda is not None else torch.zeros_like(curr_lambda)
                        mod_weights = new_lambda.detach()[5:10]  # w_en..w_img
                        L_norm = loss_dict["loss"].detach().item()
                        penalty, lyap_info = lyapunov.get_penalty(L_norm, delta_lam, mod_weights)
                        loss_dict["loss"] = loss_dict["loss"] + penalty
                        logger.log_scalar("lyapunov/V_t", lyap_info["V_t"], global_step)
                        logger.log_scalar("lyapunov/delta_V", lyap_info["delta_V"], global_step)

                    prev_lambda = curr_lambda

                # === BL baselines (EXP-013) ===

                # BL3: Uncertainty Weighting — reweight per-modality losses
                if bl_uncertainty and loss_type == "composite":
                    mod_losses = {}
                    for mod in MODALITIES:
                        key = f"loss_{mod}_contrast"
                        if key in loss_dict:
                            align_key = f"loss_{mod}_align"
                            reg_key = f"loss_{mod}_reg"
                            L_m = loss_dict[key]
                            if align_key in loss_dict:
                                L_m = L_m + loss_dict[align_key]
                            if reg_key in loss_dict:
                                L_m = L_m + loss_fn.lambda_reg * loss_dict[reg_key]
                            mod_losses[mod] = L_m
                    if mod_losses:
                        L_uw = bl_uncertainty.reweight_loss(mod_losses)
                        L_global = loss_dict.get("loss_global", torch.tensor(0.0, device=device))
                        L_va = loss_dict.get("loss_va", torch.tensor(0.0, device=device))
                        loss_dict["loss"] = loss_fn.w_g * L_global + L_uw + loss_fn.lambda_va * loss_fn._va_scale * L_va

                    # Log UW weights (same keys as controller for comparison)
                    if global_step % 10 == 0:
                        uw_weights = bl_uncertainty.get_weights()
                        uw_logvars = bl_uncertainty.get_log_vars()
                        for mod in MODALITIES:
                            logger.log_scalar(f"controller/lambda/w_{mod}", uw_weights.get(mod, 1.0), global_step)
                            logger.log_scalar(f"uncertainty/s_{mod}", uw_logvars.get(mod, 0.0), global_step)

                loss = loss_dict["loss"]

            # BL2: PCGrad — custom backward with gradient surgery
            if bl_pcgrad and loss_type == "composite":
                # Extract per-modality losses (need grad graph, so recompute from loss_dict)
                mod_losses = {}
                for mod in MODALITIES:
                    key = f"loss_{mod}_contrast"
                    if key in loss_dict:
                        align_key = f"loss_{mod}_align"
                        reg_key = f"loss_{mod}_reg"
                        w_m = loss_fn.modality_weights.get(mod, 1.0)
                        L_m = loss_dict[key]
                        if align_key in loss_dict:
                            L_m = L_m + loss_dict[align_key]
                        if reg_key in loss_dict:
                            L_m = L_m + loss_fn.lambda_reg * loss_dict[reg_key]
                        mod_losses[mod] = w_m * L_m

                L_global = loss_fn.w_g * loss_dict.get("loss_global", torch.tensor(0.0, device=device))
                L_va = loss_fn.lambda_va * loss_fn._va_scale * loss_dict.get("loss_va", torch.tensor(0.0, device=device))
                global_loss_for_pcgrad = L_global + L_va

                shared_params = [p for p in model.parameters() if p.requires_grad]
                pcgrad_stats = bl_pcgrad.step(
                    mod_losses, shared_params,
                    global_loss=global_loss_for_pcgrad,
                    scaler=scaler,
                )

                if global_step % 10 == 0:
                    logger.log_scalar("pcgrad/conflict_count", pcgrad_stats["conflict_count"], global_step)
                    logger.log_scalar("pcgrad/conflict_ratio", pcgrad_stats["conflict_ratio"], global_step)
                    logger.log_scalar("pcgrad/mean_cosine", pcgrad_stats["mean_cosine"], global_step)

                grad_norm = torch.tensor(0.0)  # PCGrad handles its own gradient clipping

            # Standard backward (for non-PCGrad)
            elif scaler:
                scaler.scale(loss).backward()
                if cfg.training.get("gradient_clip", 0) > 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.training.gradient_clip
                    )
                else:
                    grad_norm = torch.tensor(0.0)

                # BL1: GradNorm update (after backward, before optimizer step)
                if bl_gradnorm and loss_type == "composite":
                    # Need separate forward for per-modality grad norms
                    # GradNorm uses its own w_m update, not optimizer
                    mod_losses_gn = {}
                    for mod in MODALITIES:
                        key = f"loss_{mod}_contrast"
                        if key in loss_dict:
                            mod_losses_gn[mod] = loss_dict[key].detach()
                    # GradNorm step updates its internal weights
                    gn_info = bl_gradnorm.step(mod_losses_gn, model)
                    # Apply GradNorm weights to composite loss
                    gn_weights = bl_gradnorm.get_weights()
                    for mod in MODALITIES:
                        if mod in gn_weights:
                            loss_fn.modality_weights[mod] = gn_weights[mod]
                    if global_step % 10 == 0:
                        for mod in MODALITIES:
                            logger.log_scalar(f"controller/lambda/w_{mod}", gn_weights.get(mod, 1.0), global_step)
                        logger.log_scalar("gradnorm/L_grad", gn_info.get("L_grad", torch.tensor(0.0)).item(), global_step)
                        logger.log_scalar("gradnorm/G_bar", gn_info.get("G_bar", torch.tensor(0.0)).item(), global_step)

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

                # BL1: GradNorm update (non-scaler path)
                if bl_gradnorm and loss_type == "composite":
                    mod_losses_gn = {}
                    for mod in MODALITIES:
                        key = f"loss_{mod}_contrast"
                        if key in loss_dict:
                            mod_losses_gn[mod] = loss_dict[key].detach()
                    gn_info = bl_gradnorm.step(mod_losses_gn, model)
                    gn_weights = bl_gradnorm.get_weights()
                    for mod in MODALITIES:
                        if mod in gn_weights:
                            loss_fn.modality_weights[mod] = gn_weights[mod]
                    if global_step % 10 == 0:
                        for mod in MODALITIES:
                            logger.log_scalar(f"controller/lambda/w_{mod}", gn_weights.get(mod, 1.0), global_step)
                        logger.log_scalar("gradnorm/L_grad", gn_info.get("L_grad", torch.tensor(0.0)).item(), global_step)
                        logger.log_scalar("gradnorm/G_bar", gn_info.get("G_bar", torch.tensor(0.0)).item(), global_step)

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

            # Log backbone weight norms (verify backbone is training)
            if global_step % 100 == 0:
                if hasattr(model, "text_encoders"):
                    # Family A: separate encoders per modality
                    for mod in ["en", "ru", "lean", "latex"]:
                        enc = model.text_encoders[mod]
                        w_norm = sum(p.data.norm().item() for p in enc.backbone.parameters())
                        logger.log_scalar(f"weight_norm/{mod}_backbone", w_norm, global_step)
                else:
                    # Family B: single shared encoder
                    w_norm = sum(p.data.norm().item() for p in model.text_encoder.backbone.parameters())
                    logger.log_scalar("weight_norm/shared_text_backbone", w_norm, global_step)
                vis_w = sum(p.data.norm().item() for p in model.visual_encoder.parameters())
                logger.log_scalar("weight_norm/visual_encoder", vis_w, global_step)

            # Evaluation
            if global_step > 0 and global_step % eval_every == 0:
                print(f"\n[Step {global_step}] Evaluating...")
                metrics, eval_embs, eval_cents = evaluate(model, eval_loader, device, cfg.eval.retrieval_k)
                logger.log_eval_metrics(metrics, global_step)

                # Visualizations: retrieval matrix heatmap + PCA simplex + mod→centroid
                logger.log_retrieval_matrix(metrics, global_step)
                logger.log_pca_simplex(eval_embs, eval_cents, global_step, n_objects=5)
                logger.log_mod_to_centroid_heatmap(metrics, global_step)

                # Check for best model (primary: mean_crossmodal_R@k, fallback: centroid_R@k)
                k0 = cfg.eval.retrieval_k[0]
                key_metric = metrics.get(f"mean_crossmodal_R@{k0}",
                             metrics.get(f"centroid_R@{k0}", 0))
                if key_metric > ckpt_state.best_metric:
                    ckpt_state.best_metric = key_metric
                    ckpt_state.best_epoch = epoch
                    if cfg.checkpoint.get("enabled", True):
                        save_checkpoint(
                            Path(cfg.checkpoint.dir) / cfg.experiment / "best_model.pt",
                            model, optimizer, scheduler, ckpt_state, cfg, scaler,
                        )
                    print(f"  New best: mean_crossmodal_R@{k0}={key_metric:.4f}")

                # Print key metrics
                for k in cfg.eval.retrieval_k:
                    cm_key = f"mean_crossmodal_R@{k}"
                    if cm_key in metrics:
                        print(f"  {cm_key}: {metrics[cm_key]:.4f}", end="  ")
                print(f"  D_intra: {metrics.get('D_intra', 0):.4f}  collapse: {metrics.get('collapse_score', 0):.4f}")

            ckpt_state.global_step += 1

        # End of epoch
        avg_loss = epoch_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch+1} done. Avg loss: {avg_loss:.4f}")

        # Save epoch checkpoint
        ckpt_state.epoch = epoch + 1
        if cfg.checkpoint.get("enabled", True):
            save_checkpoint(
                Path(cfg.checkpoint.dir) / cfg.experiment / f"epoch_{epoch+1:03d}.pt",
                model, optimizer, scheduler, ckpt_state, cfg, scaler,
            )
            manage_checkpoints(
                Path(cfg.checkpoint.dir) / cfg.experiment,
                keep_best=cfg.checkpoint.get("keep_best", 3),
            )

        logger.flush()

    # Final evaluation on val/test
    print("\n=== Final Evaluation (val) ===")
    final_metrics, final_embs, final_cents = evaluate(model, eval_loader, device, cfg.eval.retrieval_k)
    logger.log_eval_metrics(final_metrics, ckpt_state.global_step)
    logger.log_retrieval_matrix(final_metrics, ckpt_state.global_step)
    logger.log_pca_simplex(final_embs, final_cents, ckpt_state.global_step, n_objects=5)
    logger.log_mod_to_centroid_heatmap(final_metrics, ckpt_state.global_step)

    for name, value in sorted(final_metrics.items()):
        print(f"  {name}: {value:.4f}")

    # Held-out test evaluation (if separate test set exists)
    if held_out_test_loader is not None:
        print("\n=== Held-out Test Evaluation ===")
        test_metrics, _, _ = evaluate(model, held_out_test_loader, device, cfg.eval.retrieval_k)
        logger.log_scalars("test", test_metrics, ckpt_state.global_step)
        for name, value in sorted(test_metrics.items()):
            print(f"  {name}: {value:.4f}")

    # Save final model (always, regardless of checkpoint.enabled)
    final_path = Path(cfg.checkpoint.dir) / cfg.experiment / "final_model.pt"
    save_checkpoint(final_path, model, optimizer, scheduler, ckpt_state, cfg, scaler)
    print(f"Final model saved: {final_path}")

    # Cleanup
    logger.close()
    if s3_daemon:
        s3_daemon.stop()
        print("S3 backup: final sync done")

    print(f"\nDone! Best R@{cfg.eval.retrieval_k[0]}: {ckpt_state.best_metric:.4f} (epoch {ckpt_state.best_epoch+1})")
    print(f"Logs: {logger.log_dir}")


if __name__ == "__main__":
    main()
