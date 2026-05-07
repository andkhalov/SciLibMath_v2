"""TensorBoard logger for experiment tracking.
Ref: CLAUDE.md Sec 5 — what to log (scalars, histograms).
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig, OmegaConf

from models.constants import MODALITIES


class TBLogger:
    """Wrapper around TensorBoard SummaryWriter with experiment conventions."""

    def __init__(self, cfg: DictConfig):
        exp_name = cfg.get("experiment", "unknown")
        seed = cfg.get("seed", 0)
        run_tag = cfg.get("run_tag", "")
        prefix = f"{run_tag}_" if run_tag else ""
        run_name = f"{prefix}{exp_name}_s{seed}_{int(time.time())}"

        log_dir = Path(cfg.logging.tensorboard_dir) / run_name
        self.writer = SummaryWriter(str(log_dir))
        self.log_dir = log_dir

        # Save config as text
        self.writer.add_text("config", f"```yaml\n{OmegaConf.to_yaml(cfg)}\n```", 0)

    def log_scalars(self, tag_prefix: str, scalars: dict, step: int):
        """Log multiple scalars under a prefix."""
        for name, value in scalars.items():
            if hasattr(value, "item"):
                value = value.item()
            self.writer.add_scalar(f"{tag_prefix}/{name}", value, step)

    def log_scalar(self, tag: str, value, step: int):
        if hasattr(value, "item"):
            value = value.item()
        self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag: str, values, step: int):
        self.writer.add_histogram(tag, values, step)

    def log_loss_components(self, loss_dict: dict, step: int):
        """Log all loss components from composite loss."""
        for name, value in loss_dict.items():
            if name == "loss":
                self.log_scalar("loss/total", value, step)
            elif name.startswith("loss_"):
                self.log_scalar(f"loss/{name}", value, step)
            elif name == "modality_losses":
                for mod, mloss in value.items():
                    self.log_scalar(f"loss/modality/{mod}", mloss, step)

    def log_training_stats(self, stats: dict, step: int):
        """Log training stats (lr, grad_norm, etc.)."""
        self.log_scalars("training", stats, step)

    def log_eval_metrics(self, metrics: dict, step: int):
        """Log evaluation metrics (retrieval, geometry)."""
        self.log_scalars("metrics", metrics, step)

    # --- Visualization methods ---

    def log_retrieval_matrix(self, metrics: dict, step: int):
        """Log 5×5 cross-modal retrieval heatmap for R@1 and R@10.

        Rows = query modality, columns = target modality.
        Diagonal = mod→centroid R@k (single modality embedding → full centroid gallery).
        Off-diagonal = m1_to_m2 R@k (where available; NaN if pair not computed).
        """
        for k_val in [1, 10]:
            mat = np.full((5, 5), np.nan)
            for i, m1 in enumerate(MODALITIES):
                # Diagonal: single modality → centroid retrieval
                m2c_key = f"{m1}_to_centroid_R@{k_val}"
                if m2c_key in metrics:
                    mat[i, i] = metrics[m2c_key]
                # Off-diagonal: cross-modal
                for j, m2 in enumerate(MODALITIES):
                    if i == j:
                        continue
                    cross_key = f"{m1}_to_{m2}_R@{k_val}"
                    if cross_key in metrics:
                        mat[i, j] = metrics[cross_key]

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(mat, vmin=0, vmax=1, cmap="YlOrRd", aspect="equal")
            ax.set_xticks(range(5))
            ax.set_yticks(range(5))
            ax.set_xticklabels(MODALITIES)
            ax.set_yticklabels(MODALITIES)
            ax.set_xlabel("Target")
            ax.set_ylabel("Query")
            ax.set_title(f"Cross-Modal R@{k_val} (step {step})")
            # Annotate cells
            for i in range(5):
                for j in range(5):
                    val = mat[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                                color="white" if val > 0.5 else "black", fontsize=9)
            fig.colorbar(im, ax=ax, shrink=0.8)
            fig.tight_layout()
            self.writer.add_figure(f"retrieval_matrix/R@{k_val}", fig, step)
            plt.close(fig)

    def log_mod_to_centroid_heatmap(self, metrics: dict, step: int):
        """Log 5×1 heatmap: each modality → centroid R@k (k=1 and k=10).

        Uses {mod}_to_centroid_R@k: query = single modality embedding,
        gallery = all object centroids. Match if same object.
        """
        for k_val in [1, 10]:
            vals = []
            labels = []
            for m in MODALITIES:
                key = f"{m}_to_centroid_R@{k_val}"
                if key in metrics:
                    vals.append(metrics[key])
                    labels.append(m)
            if not vals:
                continue

            arr = np.array(vals).reshape(-1, 1)  # [M, 1]
            fig, ax = plt.subplots(figsize=(2.5, 4))
            im = ax.imshow(arr, vmin=0, vmax=1, cmap="YlOrRd", aspect=0.4)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
            ax.set_xticks([0])
            ax.set_xticklabels([f"R@{k_val}"])
            ax.set_title(f"Mod→Centroid R@{k_val} (step {step})")
            for i, v in enumerate(vals):
                ax.text(0, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=10)
            fig.colorbar(im, ax=ax, shrink=0.6)
            fig.tight_layout()
            self.writer.add_figure(f"mod_to_centroid/R@{k_val}", fig, step)
            plt.close(fig)

    def log_pca_simplex(self, embeddings: dict, centroids: torch.Tensor, step: int,
                        n_objects: int = 5):
        """PCA projection of n_objects × (5 modalities + 1 centroid) → 2D scatter.

        Args:
            embeddings: {modality: [N, d]} — unnormalized embeddings from eval
            centroids: [N, d] — unnormalized centroids from eval
            step: global step
            n_objects: how many objects to visualize
        """
        n_objects = min(n_objects, centroids.shape[0])
        # Gather points: [n_objects * (M+1), d]
        points = []
        labels_mod = []  # modality label per point
        labels_obj = []  # object index per point
        for obj_idx in range(n_objects):
            for m in MODALITIES:
                points.append(embeddings[m][obj_idx])
                labels_mod.append(m)
                labels_obj.append(obj_idx)
            points.append(centroids[obj_idx])
            labels_mod.append("centroid")
            labels_obj.append(obj_idx)

        X = torch.stack(points).numpy()  # [(M+1)*n_obj, d]
        # PCA → 2D
        X_centered = X - X.mean(axis=0)
        try:
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            X_2d = X_centered @ Vt[:2].T  # [n_points, 2]
        except np.linalg.LinAlgError:
            return  # SVD failed, skip

        # Color/marker maps
        mod_colors = {"en": "#1f77b4", "ru": "#2ca02c", "lean": "#ff7f0e",
                      "latex": "#d62728", "img": "#9467bd", "centroid": "#000000"}
        obj_markers = ["o", "s", "^", "D", "v"]

        fig, ax = plt.subplots(figsize=(8, 6))
        for idx in range(len(labels_mod)):
            m = labels_mod[idx]
            o = labels_obj[idx]
            marker = obj_markers[o % len(obj_markers)]
            size = 120 if m == "centroid" else 60
            edgecolor = "black" if m == "centroid" else "none"
            marker_char = "*" if m == "centroid" else marker
            ax.scatter(X_2d[idx, 0], X_2d[idx, 1], c=mod_colors[m],
                       marker=marker_char, s=size, edgecolors=edgecolor, linewidths=1,
                       zorder=3 if m == "centroid" else 2)

        # Draw simplex edges (modalities → centroid for each object)
        for obj_idx in range(n_objects):
            cent_global = obj_idx * (len(MODALITIES) + 1) + len(MODALITIES)
            for m_idx in range(len(MODALITIES)):
                p_global = obj_idx * (len(MODALITIES) + 1) + m_idx
                ax.plot([X_2d[p_global, 0], X_2d[cent_global, 0]],
                        [X_2d[p_global, 1], X_2d[cent_global, 1]],
                        color="gray", alpha=0.3, linewidth=0.8, zorder=1)

        # Legend
        for m, c in mod_colors.items():
            mk = "*" if m == "centroid" else "o"
            ax.scatter([], [], c=c, marker=mk, s=60, label=m)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"PCA: {n_objects} objects × 5 modalities + centroid (step {step})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        fig.tight_layout()
        self.writer.add_figure("pca/simplex_5objects", fig, step)
        plt.close(fig)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()
