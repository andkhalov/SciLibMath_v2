# TZ.md — Техническое задание SciLibMath_v2

> **Версия:** 1.6 | **Фаза:** 4 | **Дата:** 2026-03-09
> **Привязка:** MATH.md v2.6 (формализация), v_1/ (реализация baseline)

---

## 0. Репозиторий и инфраструктура

### 0.1 Структура кода

```
SciLibMath_v2/
├── v_2/
│   ├── writing/          # Документация (MOTIVATION, LIT, MATH, TZ)
│   ├── src/
│   │   ├── data/         # Датасет, загрузчик, коллатор
│   │   ├── models/       # Архитектура энкодеров (Family A, B)
│   │   ├── losses/       # E1-E7 функции потерь
│   │   ├── controller/   # T-S fuzzy controller (E6, E7)
│   │   ├── metrics/      # Метрики (R@k, collapse, balance, etc.)
│   │   ├── train/        # Training loop, checkpointing
│   │   └── config/       # Конфигурация экспериментов
│   ├── experiments/      # Результаты E1-E7
│   ├── tokenizers/       # Обученные токенизаторы Lean, LaTeX
│   └── requirements.txt
├── v_1/                  # Зафиксированная первая версия
└── literature/           # Литература
```

### 0.2 Зависимости

```
# Core
torch>=2.1
transformers>=4.36
tokenizers>=0.15

# Data
datasets>=2.16        # HuggingFace datasets
Pillow>=10.0          # Image processing

# Visual encoder
torchvision>=0.16     # ResNet

# Metrics & logging
tensorboard>=2.15     # или wandb>=0.16
scikit-learn>=1.3     # retrieval metrics

# Fuzzy controller (E6, E7)
numpy>=1.24
scipy>=1.11           # для LMI (если нужно)
```

### 0.3 Вычислительные требования

```
Минимум: 1× GPU с 24GB VRAM (RTX 3090 / A5000)
Рекомендовано: 1× GPU с 40-80GB VRAM (A100)

Причина: 5 энкодеров Family A ≈ 80M параметров.
Batch size 32-64, mixed precision (fp16/bf16).

Альтернатива: gradient accumulation для меньших GPU.
```

---

## 1. Датасет: SciLibModal_v2

### 1.1 Источник

```
GitHub: https://github.com/andkhalov/scilibrumodal-v2

Описание: ~1M записей, каждая содержит 5 модальностей:
- en: описание на английском
- ru: описание на русском
- lean: формальная запись Lean 4
- latex: LaTeX-нотация
- img: визуальное представление формулы (PNG, h=64px, variable width)

Составлен из: HERALD + ATLAS + MathLib4 (Jixia + LLM-augmentation)
```

### 1.2 Загрузчик данных

```python
# Реализация: src/data/dataset.py
# Ref: v_1/FLOM_train_ALL_personal_loss.ipynb, Cell 9 (FlomTorchDataset)

class SciLibModalDataset(Dataset):
    """
    Мультимодальный датасет.

    Для каждого объекта возвращает dict:
        {
            'en': str,
            'ru': str,
            'lean': str,
            'latex': str,
            'img': PIL.Image или None  # если нет визуального представления
        }

    [Assumption A.1] Все 5 модальностей должны быть доступны.
    Объекты без полного набора — отфильтровываются при загрузке.
    """
```

### 1.3 Коллатор

```python
# Реализация: src/data/collator.py
# Ref: v_1/FLOM_train_ALL_personal_loss.ipynb, Cell 9 (FlomCollator)

class SciLibModalCollator:
    """
    Токенизация и padding.

    Входы: list of dicts (из SciLibModalDataset)
    Выходы: dict of tensors:
        {
            'en': {'input_ids': [B,L], 'attention_mask': [B,L]},
            'ru': {'input_ids': [B,L], 'attention_mask': [B,L]},
            'lean': {'input_ids': [B,L], 'attention_mask': [B,L]},
            'latex': {'input_ids': [B,L], 'attention_mask': [B,L]},
            'img': [B, K_max, 3, 64, 64]  # K_max = max overlapping patches in batch
            # K = floor((W - patch_size) / stride) + 1, stride=32, patch_size=64
        }
    """
```

---

## 2. Архитектура — Family A [Paper A]

```python
# Реализация: src/models/family_a.py
# Ref: MATH.md M.1.1, M.1.2
# Ref: v_1/FLOM_train_ALL_personal_loss.ipynb, Cell 16 (MultiModalHeraldEncoder)

class FamilyAEncoder(nn.Module):
    """
    5 отдельных энкодеров + shared projection.

    Компоненты:
    - enc_en, enc_ru, enc_lean, enc_latex: Transformer (SciRus-tiny 3.5 zh)
    - enc_img: VisualEncoder (ResNet→AlignLayer→SciRus-tiny, см. секция 5)
    - proj_en, proj_ru, proj_lean, proj_latex: Linear(d', d)
    - proj_img: встроен в VisualEncoder

    Forward:
        batch -> dict of embeddings {m: [B, d]}

    Параметры: ~80M

    # All layers trainable (full fine-tuning, M.2.4 DEPRECATED unfreezing)
    """

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        embeddings = {}
        for m in ['en', 'ru', 'lean', 'latex']:
            h = self.encoders[m](batch[m])  # [B, L, d']
            h = h.last_hidden_state.mean(dim=1)  # mean pooling -> [B, d']
            embeddings[m] = self.projections[m](h)  # [B, d]

        embeddings['img'] = self.enc_img(batch['img'])  # [B, d]
        return embeddings
```

---

## 3. Архитектура — Family B [Paper A]

> **Family B — чистый ablation baseline.** Все весовые коэффициенты (w_g, w_m, λ_align, λ_rad, λ_reg, λ_va) фиксированы перед обучением. Fuzzy controller (Sec 7) и LossMixer (E5) НЕ применяются. Единственная переменная — архитектура энкодеров. E1-E4 на обоих Family A и B. E5-E7 — только Family A. (Ref: MATH.md M.1.3)

```python
# Реализация: src/models/family_b.py
# Ref: MATH.md M.1.3

class FamilyBEncoder(nn.Module):
    """
    1 symbolic encoder + 1 visual encoder.

    Компоненты:
    - enc_sym: единый Transformer для {en, ru, lean, latex}
    - enc_img: ResNetVisualEncoder
    - proj_sym: Linear(d', d)
    - proj_img: Linear(d'_img, d)

    Модальность определяется через детерминированный onehot вектор-индекс:
        e ∈ R^d — семантический эмбеддинг (для лоссов и центроидов)
        ẽ = [e; onehot(m)] ∈ R^{d+M} — полное представление (для retrieval)
        (MATH.md M.1.3)

    Параметры: ~28M
    """

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        embeddings = {}       # e ∈ R^d — для лоссов и центроидов
        full_repr = {}        # ẽ ∈ R^{d+M} — с onehot для идентификации

        for m in ['en', 'ru', 'lean', 'latex']:
            h = self.enc_sym(batch[m])
            h = h.last_hidden_state.mean(dim=1)
            e = self.proj_sym(h)                      # R^d
            embeddings[m] = e

            # Детерминированный onehot — НЕ обучаемый, для идентификации
            onehot = self.modality_onehot[m].expand(e.size(0), -1)
            full_repr[m] = torch.cat([e, onehot], dim=-1)  # R^{d+M}

        e_img = self.enc_img(batch['img'])             # R^d
        embeddings['img'] = e_img
        onehot_img = self.modality_onehot['img'].expand(e_img.size(0), -1)
        full_repr['img'] = torch.cat([e_img, onehot_img], dim=-1)

        return embeddings  # лоссы работают с R^d, onehot хранится отдельно
```

---

## 4. Токенизаторы Lean и LaTeX [Paper A]

```python
# Реализация: src/data/tokenizer_training.py
# Ref: MATH.md M.2

# Обучение BPE-токенизаторов из корпуса SciLibModal_v2

from tokenizers import Tokenizer, models, trainers, pre_tokenizers

def train_lean_tokenizer(corpus_path: str, vocab_size: int = 16000):
    """
    Обучает BPE-токенизатор для Lean 4.

    Особенности:
    - Pre-tokenization: split по whitespace + unicode категориям
    - Special tokens: [PAD], [UNK], [CLS], [SEP], [MASK]
    - Lean-специфичные: сохранение Unicode символов (→, ∀, ∃, etc.)
    """

def train_latex_tokenizer(corpus_path: str, vocab_size: int = 16000):
    """
    Обучает BPE-токенизатор для LaTeX.

    Особенности:
    - Pre-tokenization: split по backslash (\\) для сохранения команд
    - \\frac, \\sum, \\int — один токен каждый
    - {, }, ^, _ — отдельные токены
    """

def initialize_new_tokens_fvt(base_tokenizer, domain_tokenizer, base_embeddings):
    """
    FVT (Fast Vocabulary Transfer) — инициализация эмбеддингов новых токенов.
    Ref: MATH.md M.2.3

    Для каждого нового токена t_new ∈ V_domain \\ V_base:
    1. Декомпозиция: D(t_new) = [t_1, ..., t_k] — разбиение под base_tokenizer
    2. Инициализация: e_{t_new} = mean(e_{t_1}, ..., e_{t_k})

    Пример: \\frac → base разбивает на ['\\\\', 'fr', 'ac']
    e_{\\frac} = (e_{\\\\ } + e_{fr} + e_{ac}) / 3
    """

# Стратегия обучения (Ref: MATH.md M.2.4)
# DEPRECATED: трёхфазный unfreezing убран (SciRus-tiny 3 слоя + FVT = full fine-tuning).
# Discriminative LR: backbone lr = lr * lr_embed_ratio (default 0.1)
```

---

## 5. Визуальный поток (ResNet $\to$ AlignNet $\to$ SciRus-tiny) [Paper A]

```python
# Реализация: src/models/visual_encoder.py
# Ref: MATH.md M.1.2, M.1.2*

class VisualEncoder(nn.Module):
    """
    Visual encoder для формул переменной длины.

    Pipeline (согласован с MATH.md M.1.2 v2.4):
    1. Разбиение изображения на overlapping патчи 64×64, stride=32
    2. ResNet feature extraction для каждого патча → визуальные токены
    3. AlignNet: LayerNorm(Linear(d_vis, d_text)) — нормализующая проекция
    4. Подача последовательности в SciRus-tiny (как текстовые токены)
    5. Mean pooling по позициям (устраняет избыточность от overlap)
    6. Linear projection в R^d

    Изменение v2.4: overlapping patches (stride=p/2=32).
    Граничные признаки сохраняются — каждый пиксель в полосе
    перекрытия (32px) покрывается двумя соседними патчами.
    Ref: PVTv2 (Wang 2022), T2T-ViT (Yuan 2021), CvT (Wu 2021).

    Ref: SETUP_DRAFT.md "ResNet будет давать вектор визуальных токенов...
         далее мы делаем секвенцию таких визуальных символов и в
         нормированном виде (через выравнивающий слой) подаем в SciRus-tiny"
    """

    def __init__(self, d_embed: int = 256, d_text: int = 312,
                 patch_size: int = 64, stride: int = 32,
                 resnet_variant: str = 'resnet18'):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride  # s = p/2 → overlapping patches (MATH.md M.1.2)

        resnet = torchvision.models.resnet18(pretrained=True)
        # Убираем последний FC и avg_pool → feature extractor
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # AlignNet: LayerNorm(Linear(d_vis, d_text))
        d_vis = 512  # resnet18 output channels
        self.align_net = nn.Sequential(
            nn.Linear(d_vis, d_text),
            nn.LayerNorm(d_text)
        )

        # SciRus-tiny для обработки последовательности визуальных токенов
        self.text_encoder = None  # устанавливается из FamilyAEncoder

        # Projection head
        self.proj = nn.Linear(d_text, d_embed)

    def extract_patches(self, img: torch.Tensor) -> torch.Tensor:
        """
        Разбиение изображения на overlapping патчи.
        img: [B, 3, H, W] — H=64, W=variable
        returns: [B, K, 3, patch_size, patch_size]
        K = floor((W - patch_size) / stride) + 1
        """
        B, C, H, W = img.shape
        p, s = self.patch_size, self.stride
        # unfold по ширине с overlapping stride
        patches = img.unfold(3, p, s)          # [B, 3, H, n_w, p]
        patches = patches.permute(0, 3, 1, 2, 4)  # [B, n_w, 3, H, p]
        return patches  # [B, K, 3, 64, 64]

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        patches: [B, K, 3, 64, 64] — K overlapping патчей
        returns: [B, d_embed]
        """
        B, K, C, H, W = patches.shape

        # Шаг 1-2: ResNet feature extraction
        x = patches.view(B * K, C, H, W)     # [B*K, 3, 64, 64]
        x = self.resnet_features(x)            # [B*K, 512, h', w']
        x = self.adaptive_pool(x)              # [B*K, 512, 1, 1]
        x = x.view(B, K, -1)                  # [B, K, 512]

        # Шаг 3: AlignNet (Linear + LayerNorm)
        x = self.align_net(x)                  # [B, K, d_text]

        # Шаг 4: SciRus-tiny (обработка как последовательность)
        h = self.text_encoder(inputs_embeds=x)  # [B, K, d_text]
        h = h.last_hidden_state                 # [B, K, d_text]

        # Шаг 5: Mean pooling (устраняет избыточность от overlap)
        h = h.mean(dim=1)                       # [B, d_text]

        # Шаг 6: Projection
        return self.proj(h)                     # [B, d_embed]

    def get_aligned_tokens(self, patches: torch.Tensor) -> torch.Tensor:
        """Возвращает визуальные токены после AlignNet (для L_visual_align)."""
        B, K, C, H, W = patches.shape
        x = patches.view(B * K, C, H, W)
        x = self.resnet_features(x)
        x = self.adaptive_pool(x)
        x = x.view(B, K, -1)
        return self.align_net(x)              # [B, K, d_text]


class VisualAlignLoss(nn.Module):
    """
    Auxiliary contrastive loss: визуальные pooled embed ↔ LaTeX pooled embed.
    Ref: MATH.md M.1.2* (L_visual_align)
    Ref: Hadsell, Chopra, LeCun (2006) contrastive loss

    В E1-E5: λ_va = const (статический гиперпараметр).
    В E6-E7: λ_va — компонент λ_t, управляемый fuzzy controller (правило R6, M.6.3).
    Добавляется ко всем E1-E7 (MATH.md M.3.0).

    Размерности: visual_pooled ∈ R^{d_text}, latex_pooled ∈ R^{d_text}
    (обе d_text после AlignNet / SciRus-tiny embedding layer).
    """

    def __init__(self, margin: float = 1.0, lambda_va: float = 0.1):
        super().__init__()
        self.margin = margin      # δ_va — margin для негативных пар
        self.lambda_va = lambda_va

    def forward(self, visual_pooled: torch.Tensor,
                latex_pooled: torch.Tensor) -> torch.Tensor:
        """
        visual_pooled: [B, d_text] — mean-pooled визуальные токены после AlignNet
        latex_pooled: [B, d_text] — mean-pooled LaTeX embeddings

        returns: scalar loss
        """
        B = visual_pooled.size(0)

        # Позитивные пары: ||v_i - t_i||^2
        pos_loss = (visual_pooled - latex_pooled).pow(2).sum(dim=1).mean()

        # Негативные пары: max(0, margin - ||v_i - t_j||)
        # Для эффективности: in-batch negatives
        dists = torch.cdist(visual_pooled, latex_pooled)  # [B, B]
        mask = ~torch.eye(B, dtype=bool, device=dists.device)
        neg_dists = dists[mask].view(B, B - 1)
        neg_loss = torch.relu(self.margin - neg_dists).mean()

        return self.lambda_va * (pos_loss + neg_loss)
```

---

## 6. Experiments E1-E7 [Paper B, C]

### 6.1 Общая структура

```python
# Реализация: src/losses/

# Каждый эксперимент — отдельный модуль:
# src/losses/e1_pairwise_infonce.py
# src/losses/e2_centroid_infonce.py
# src/losses/e3_centroid_full.py
# src/losses/e4_composite_static.py
# src/losses/e5_composite_learnable.py
# src/losses/e6_fuzzy_controller.py
# src/losses/e7_lyapunov_regularized.py

# Общий интерфейс:
class BaseLoss(nn.Module):
    def forward(self, embeddings: Dict[str, Tensor]) -> Tuple[Tensor, Dict]:
        """
        embeddings: {modality: [B, d]}
        returns: (loss_scalar, stats_dict)
        """
```

### 6.2 E1: Pairwise InfoNCE

```python
# src/losses/e1_pairwise_infonce.py
# Ref: MATH.md M.3.1

def pairwise_infonce(embeddings, tau=0.1):
    """
    L_E1 = (1/C(M,2)) Σ_{m<m'} L_pair^{m,m'}

    10 попарных InfoNCE для M=5.
    """
```

### 6.3 E2: Centroid InfoNCE

```python
# src/losses/e2_centroid_infonce.py
# Ref: MATH.md M.3.2

def centroid_infonce(embeddings, tau=0.1, p_drop=0.3):
    """
    L_E2 = (1/N) Σ_i L_centroid(i)

    С modality dropout для создания пертурбированного центроида.
    """
```

### 6.4 E3-E5: Composite losses

```python
# src/losses/e3_centroid_full.py    — L_contrast + L_align + L_rad + L_reg
# src/losses/e4_composite_static.py — E3 + personal losses, W = const
# src/losses/e5_composite_learnable.py — E4 + LossMixer
# Ref: MATH.md M.3.3-M.3.5
```

### 6.5 E6: Fuzzy controller

```python
# src/losses/e6_fuzzy_controller.py
# src/controller/fuzzy_ts.py
# Ref: MATH.md M.3.6, M.6

# L_E6 = w_{g,t} · L_E3(λ_t) + Σ_m w_{m,t} · L_personal^m(λ_t)
# где w_{g,t}, w_{m,t} — компоненты λ_t, управляемые fuzzy controller.
# В отличие от E5 (LossMixer, MLP через backprop),
# E6 использует символьные правила (M.6.3) для корректировки λ_t.
```

### 6.6 E7: Lyapunov regularized

```python
# src/losses/e7_lyapunov_regularized.py
# src/controller/lyapunov.py
# Ref: MATH.md M.3.7, M.7
```

### 6.7 E8: Nonlinear T-S controller

```python
# src/controller/rules.py — NonlinearConsequent class
# src/controller/ts_controller.py — nonlinear_consequents=True option
# Ref: MATH.md M.3.8

# E8 = E6 but with MLP consequents instead of linear A_r·s_t + b_r.
# φ_r(s_t) = W2·ReLU(W1·s_t + b1) + b2, per rule.
# Antecedents (MF, t-norm, normalization) unchanged.
# Config: controller.nonlinear_consequents=true, controller.consequent_hidden=32
```

### 6.8 E9: Potential Loss + LossMixer

```python
# src/losses/potential.py — PotentialLoss class
# Ref: MATH.md M.3.9

# L_potential = k_a · attract + k_r · repel
# attract = (1/NM) Σ_i Σ_m ||e_m^i - c_i||²  (harmonic)
# repel = -(1/C(N,2)) Σ_{i<j} log(||c_i - c_j|| + ε)  (log-barrier)
# Replaces L_align + L_rad in composite loss.
# E9 = L_contrast(E2) + L_potential + L_ac + LossMixer (E5 style)
# Config: loss.use_potential=true, loss.k_a=1.0, loss.k_r=0.1
```

### 6.9 E10: Potential Loss + Fuzzy controller

```python
# Ref: MATH.md M.3.10

# E10 = L_contrast(E2) + L_potential + L_ac + L_va + fuzzy controller
# Same as E6, but L_potential replaces L_align + L_rad.
# Controller manages: [τ, k_a, k_r, λ_reg, λ_va, w_en, ..., w_g]
# Config: loss.use_potential=true + controller section
```

---

## 7. Fuzzy Controller: реализация T-S [Paper C]

```python
# Реализация: src/controller/fuzzy_ts.py
# Ref: MATH.md M.6

class FuzzyTSController:
    """
    Takagi-Sugeno fuzzy controller для управления гиперпараметрами.

    Компоненты:
    - MembershipFunctions: определяют лингвистические переменные
    - RuleBase: 7 правил R0-R6 (M.6.3)
    - Defuzzifier: агрегация выходов правил (product t-norm)

    Interface:
        s_t -> u_t
        где s_t ∈ R^18 (M.5.1):
            Группа 1 (агр., 8): L, ΔL, EMA, Var, conflict, collapse,
                                 gradient_norm, retrieval_drift
            Группа 2 (per-m loss, 5): L_en, L_ru, L_lean, L_latex, L_img
            Группа 3 (per-m EMA, 5): EMA_en, EMA_ru, EMA_lean, EMA_latex, EMA_img
            u_t ∈ R^11 (корректировка λ_t: τ, λ_align, λ_rad, λ_reg, λ_va,
                         w_en, w_ru, w_lean, w_latex, w_img, w_g)

    Правила:
        R0: Fallback (всё нормально → нет корректировок)
        R1: Modality Imbalance (перебалансировка w_m)
        R2: Loss Plateau (увеличение τ)
        R3: Dominant Modality (снижение w доминантной)
        R4: Collapse Risk (усиление λ_reg)
        R5: Gradient Conflict (снижение τ, увеличение λ_align)
        R6: Visual Misalignment (усиление λ_va, L_{img,t} как proxy)
    """

    def __init__(self, config: FuzzyConfig):
        self.membership_fns = self._build_membership_fns(config)
        self.rules = self._build_rules(config)

    def compute_control(self, state: np.ndarray) -> np.ndarray:
        """
        1. Вычислить степени принадлежности для каждого компонента s_t
        2. Активировать правила (product t-norm для антецедентов)
        3. Нормализовать активации
        4. Вычислить взвешенный выход u_t = Σ h̄_r · (A_r·s_t + b_r)
        """

    def update_hyperparams(self, lambda_t, u_t, bounds):
        """
        Variant D (MATH.md M.6.3b):
        λ_{t+1} = Π_Λ(λ_t + α·u_t + γ·(λ_0 - λ_t))

        Parameters (from config):
          alpha: 0.001       # step size (reduced from 0.05)
          warmup_steps: 200  # raw state for first 200 steps
          step_frequency: 10 # apply every K=10 steps
          noise_sigma: 0.01  # stochastic exploration σ_0
          noise_anneal: true # linear σ decay to 0
          elastic_gamma: 0.01 # mean-reversion coefficient γ
          bounds: w_m ∈ [0.3, 3.0] (narrowed from [0.1, 5.0])
        """

class MembershipFunction:
    """Треугольная функция принадлежности."""
    def __init__(self, center, width):
        self.center = center
        self.width = width

    def __call__(self, x):
        return max(0, 1 - abs(x - self.center) / self.width)
```

---

## 8. Lyapunov regularization [Paper C]

```python
# Реализация: src/controller/lyapunov.py
# Ref: MATH.md M.7

class LyapunovRegularizer:
    """
    Функция Ляпунова как soft constraint.

    V_t = α·L̃_t + β·||Δλ_t||² + γ·Var_m(w_{m,t})

    # Soft constraint (MATH.md M.7.2b):
    L_lyap = penalty_weight · max(0, ΔV_t - ξ)
    # where ΔV_t = V_t - V_{t-1}, ξ = allowable growth threshold (default 0.01)
    # Penalizes only GROWTH beyond ξ, not absolute V_t value
    """

    def __init__(self, alpha=1.0, beta=0.1, gamma=0.1,
                 eta=0.01, xi=0.001, lambda_lyap=0.1):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.xi = xi
        self.lambda_lyap = lambda_lyap
        self.V_prev = None

    def compute_V(self, L_smooth, delta_lambda, weight_variance):
        return (self.alpha * L_smooth +
                self.beta * (delta_lambda ** 2).sum() +
                self.gamma * weight_variance)

    def compute_loss(self, V_current, psi_t):
        if self.V_prev is None:
            self.V_prev = V_current.detach()
            return torch.tensor(0.0)

        violation = V_current - self.V_prev + self.eta * psi_t - self.xi
        loss = self.lambda_lyap * torch.relu(violation)
        self.V_prev = V_current.detach()
        return loss
```

---

## 9. Метрики [Paper A, B, C]

```python
# Реализация: src/metrics/

class RetrievalMetrics:
    """
    Centroid Recall@1, @10:
        Для query модальности m: кодируем → ищем ближайший центроид
        → проверяем совпадение объекта.

    Direct Cross-Modal Recall@1, @10:
        Для пары (m, m'): кодируем query в m → ищем ближайший e^{m'}
        → проверяем совпадение.
    """

class GeometricMetrics:
    """
    Mean Intra-Object Radius:
        D_intra = (1/N) Σ_i (1/M) Σ_m ||e_i^m - c_i||²

    Inter-Centroid Margin:
        D_inter = (1/C(N,2)) Σ_{i<j} ||c_i - c_j||²
    """

class StabilityMetrics:
    """
    Collapse Score:
        CS = clamp((cos_neg - 0.1) / 0.9, 0, 1)

    Modality Balance Score:
        MBS = 1 - std(L_m) / mean(L_m)  # 1 = perfect balance

    Variance Across Seeds:
        Запуск с 3+ seeds, вычисление std метрик.

    Time-to-Threshold:
        Шагов до достижения target R@1 (например 0.5).
    """

class DownstreamMetrics:
    """
    Theorem Retrieval:
        Query: NL описание теоремы → retrieve Lean statement

    RAG ATP Success:
        Query → retrieve premises → proof search → success/fail
    """
```

### 9.1 Полный список метрик

| Метрика | Формула | Для Paper |
|---|---|---|
| **mean_crossmodal_R@1** | **Mean R@1 across all 20 directed pairs — PRIMARY METRIC** | **A, B, C** |
| mean_crossmodal_R@{3,10} | Mean R@k across all 20 pairs | A, B |
| Centroid R@1 (LOO) | Leave-one-out centroid retrieval (NOTE: structurally saturates at ~1.0 due to 4/5 overlap — use as sanity check only) | A, B |
| Direct R@1 (m→m') | Cross-modal top-1, all M*(M-1)=20 directed pairs | A, B |
| Direct R@10 (m→m') | Cross-modal top-10, all 20 pairs | A, B |
| D_intra | Mean intra-object radius | B |
| D_inter | Mean inter-centroid distance | B |
| Collapse Score (CS) | 0=healthy, 1=collapsed | B, C |
| Modality Balance (MBS) | 1=balanced, 0=imbalanced | B, C |
| Variance across seeds | std of metrics across 3+ seeds | C |
| Time-to-threshold | Steps to R@1 ≥ 0.5 | C |
| Downstream ATP | Proof search success rate | A, B |

---

## 10. Training loop, checkpointing, воспроизводимость

### 10.1 Training loop

```python
# Реализация: src/train/trainer.py
# Ref: v_1/FLOM_train_ALL_personal_loss.ipynb, последние ячейки

class Trainer:
    """
    Основные параметры:
    - optimizer: AdamW
    - scheduler: CosineAnnealingLR с warmup
    - mixed precision: torch.cuda.amp
    - gradient clipping: max_norm=1.0

    Для E6/E7:
    - fuzzy controller вызывается каждые N_ctrl шагов
    - Lyapunov V_t вычисляется на каждом шаге

    Checkpointing:
    - Сохранение каждые N_save шагов
    - Best model по validation loss
    - Полный state: model, optimizer, scheduler, step, loss_history,
      fuzzy_state (для E6/E7)
    """
```

### 10.2 Конфигурация экспериментов

```yaml
# src/config/experiment.yaml

experiment:
  name: "E3_centroid_full"
  seed: 42
  family: "A"  # or "B"

data:
  dataset: "scilibrumodal-v2"
  batch_size: 64
  max_length: 128
  num_workers: 4

model:
  d_embed: 256
  text_backbone: "mlsa-iai-msu-lab/sci-rus-tiny3.5-zh"
  visual_backbone: "resnet18"
  visual_patch_size: 64
  visual_stride: 32          # s = p/2 → overlapping patches (MATH.md M.1.2)
  lr_embed_ratio: 0.1  # discriminative LR for backbone

loss:
  variant: "E3"  # E1-E7
  tau: 0.1
  lambda_align: 1.0
  lambda_rad: 0.1      # radial regularization (M.3.3)
  rho: 0.1             # target radius for L_rad
  lambda_reg: 0.1
  p_drop: 0.3
  align_clip: 10.0
  lambda_va: 0.1       # visual alignment loss weight (static в E1-E5; dynamic в E6-E7 через fuzzy R6)
  visual_align_margin: 1.0  # δ_va — margin для негативных пар в L_visual_align

# Только для E6/E7 (Variant D — MATH.md M.6.3b):
controller:
  alpha: 0.001          # step size (reduced from 0.05 — EXP-001 fix)
  warmup_steps: 200     # raw state during warmup
  step_frequency: 10    # apply controller every K steps
  noise_sigma: 0.01     # stochastic exploration σ_0
  noise_anneal: true    # linear annealing σ_t = σ_0·(1-t/T)
  elastic_gamma: 0.01   # mean-reversion to λ_0
  # Bounds: w_m ∈ [0.3, 3.0] (narrowed from [0.1, 5.0])

# Только для E7:
lyapunov:
  alpha: 1.0
  beta: 0.1
  gamma: 0.5
  penalty_weight: 0.1
  xi: 0.01             # allowable V growth threshold (M.7.2b)

training:
  epochs: 10
  lr: 1e-4
  weight_decay: 0.01
  warmup_steps: 500
  max_grad_norm: 1.0
  fp16: true
  save_every: 1000
  eval_every: 500

logging:
  backend: "tensorboard"  # или "wandb"
  log_dir: "experiments/{name}/logs"
```

### 10.3 Воспроизводимость

```
Для каждого эксперимента:
1. Фиксированный seed (torch, numpy, random)
2. Deterministic training (torch.backends.cudnn.deterministic = True)
3. Сохранение полного конфига в experiments/{name}/config.yaml
4. Git hash коммита в metadata
5. Минимум 3 seed runs для variance metrics
```

---

## 11. План реализации (приоритеты)

### Phase 1: Data Pipeline
1. Загрузить SciLibModal_v2 из GitHub
2. Реализовать SciLibModalDataset + SciLibModalCollator
3. Обучить токенизаторы Lean и LaTeX
4. Тест: загрузка батча, проверка размерностей

### Phase 2: Baseline Models
5. Реализовать FamilyAEncoder
6. Реализовать ResNetVisualEncoder
7. Реализовать E1 (pairwise InfoNCE)
8. Тест: forward pass, backward pass, один эпох

### Phase 3: Core Experiments
9. Реализовать E2 (centroid InfoNCE)
10. Реализовать E3 (centroid + align + reg)
11. Реализовать E4 (static weights)
12. Реализовать E5 (LossMixer — портировать из v_1)
13. Реализовать метрики
14. Запуск E1-E5 (Family A)
15. Реализовать FamilyBEncoder
16. Запуск E1-E5 (Family B)

### Phase 4: Fuzzy + Lyapunov
17. Реализовать FuzzyTSController
18. Реализовать LyapunovRegularizer
19. Реализовать E6 (fuzzy)
20. Реализовать E7 (fuzzy + Lyapunov)
21. Запуск E6-E7 (Family A)

### Phase 5: Analysis
22. Сравнительная таблица E1-E7 × Family A/B
23. Ablation analysis
24. Variance across seeds (3 runs)
25. Downstream: theorem retrieval benchmark

---

╔══════════════════════════════════════════════════════════════╗
║  META-REASONING LOOP                                         ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  [MR.1] КАК МЫ ПРИШЛИ К ЭТОМУ ДОКУМЕНТУ                     ║
║                                                              ║
║  Исходные данные: MATH.md v2.3 (формализация), v_1 ноутбук   ║
║  (реализация), CLAUDE.md Sec. 4 Phase 4 (структура TZ).      ║
║                                                              ║
║  Изменения v1.5 → v1.6 (каскад от MATH.md v2.6):           ║
║  - Sec 4: FVT initialization + unfreezing config              ║
║  - Sec 5.5: λ_va static E1-E5, dynamic E6-E7 (R6)           ║
║  - Sec 7: u_t R^10 → R^11, 5→7 правил (+R6 Visual Misalign) ║
║  - Sec 3: Family B clarification (no fuzzy, pure ablation)    ║
║  - Config: lambda_va dynamic note для E6/E7                   ║
║                                                              ║
║  Предыдущие изменения (v1.4 → v1.5, MATH.md v2.5):          ║
║  - Sec 7: FuzzyTSController docstring s_t R^8 → R^18         ║
║  - Sec 6.5: E6 формула с w_{g,t}, w_{m,t}                   ║
║  - MR.3: s_t ∈ R^18, u_t ∈ R^10 (v1.5)                      ║
║                                                              ║
║  Предыдущие изменения (v1.3 → v1.4):                        ║
║  - Sec 5: non-overlapping → overlapping patches               ║
║    (stride=32, patch_size=64, extract_patches method)          ║
║  - Collator: K_max учитывает overlap                          ║
║  - Config: visual_patch_size, visual_stride                    ║
║                                                              ║
║  Предыдущие изменения (v1.2 → v1.3):                        ║
║  - AlignNet, L_visual_align, lambda_va                        ║
║                                                              ║
║  [MR.3] ЕСТЬ ЛИ ПРОТИВОРЕЧИЯ                                ║
║                                                              ║
║  Уровень 1: Код соответствует формулам из MATH.md v2.4.      ║
║  Уровень 2: Портирование LossMixer из v_1 (E5) —             ║
║    согласовано с MATH.md M.3.5.                               ║
║    Visual encoder: overlapping patches→ResNet→AlignNet→       ║
║    SciRus-tiny → согласовано с MATH.md M.1.2 v2.4.            ║
║    L_visual_align: согласовано с MATH.md M.1.2* и M.3.0.     ║
║    s_t ∈ R^18, u_t ∈ R^11 → согласовано с M.5.1, M.3.6 v2.6.║
║    λ_t ∈ R^11 (λ_va в fuzzy E6-E7, R6), 7 правил R0-R6.     ║
║                                                              ║
║  [MR.4] ПРОВЕРКА НА ГАЛЛЮЦИНАЦИИ                            ║
║                                                              ║
║  G2: resnet18 → 512 features — корректно (torchvision docs). ║
║  G2: AlignNet(512→d_text) → SciRus-tiny(d_text) ✓            ║
║  G2: VisualAlignLoss: visual_pooled ∈ R^{d_text},             ║
║      latex_pooled ∈ R^{d_text} — размерности совпадают ✓      ║
║                                                              ║
║  [MR.5] ПАТТЕРНЫ                                             ║
║  [P.9] ALIGNMENT LAYER — применён (AlignNet в M.1.2).        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
