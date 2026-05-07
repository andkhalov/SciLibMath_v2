"""Custom tokenizers for Lean and LaTeX modalities.
Ref: MATH.md M.2.3 — Vocabulary Extension + FVT initialization.

Approach:
  1. Train BPE on domain corpus (lean / latex from SciLibModal_v2)
  2. V_new = V_domain \\ V_base  (new domain-specific tokens)
  3. Extend base tokenizer (SciRus-tiny) with new tokens
  4. Resize model embedding layer
  5. FVT init: e_new = mean(e_sub_1, ..., e_sub_k)
"""

import logging
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def extract_domain_corpus(data_dir: str | Path, field: str, max_samples: int = 0) -> list[str]:
    """Extract text corpus from HuggingFace dataset for BPE training.

    Args:
        data_dir: path to scilibrumodal-v2-data directory (HF datasets format)
        field: column name ('formal_statement' for lean, 'formula_md' for latex)
        max_samples: if > 0, limit corpus size (0 = use all)
    Returns:
        list of strings (one per sample)
    """
    from datasets import load_from_disk

    dsd = load_from_disk(str(data_dir))
    ds = dsd["train"]

    if max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    corpus = ds[field]

    # Filter empty/None
    corpus = [t for t in corpus if t and len(t.strip()) > 0]

    logger.info(f"Extracted {len(corpus)} samples for field '{field}'")
    return corpus


def train_domain_bpe(corpus: list[str], vocab_size: int = 16000) -> Tokenizer:
    """Train a BPE tokenizer on domain corpus.

    Uses ByteLevel pre-tokenizer to preserve Unicode (Lean: →, ∀, ∃, ∧, ∨, ¬, ≤;
    LaTeX: \\frac, \\sum, \\int, etc.)

    Args:
        corpus: list of domain texts
        vocab_size: target vocabulary size
    Returns:
        trained HuggingFace Tokenizer
    """
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        show_progress=True,
    )

    tokenizer.train_from_iterator(corpus, trainer=trainer)
    logger.info(f"Trained BPE tokenizer: vocab_size={tokenizer.get_vocab_size()}")
    return tokenizer


def get_new_tokens(domain_bpe: Tokenizer, base_tokenizer: AutoTokenizer) -> list[str]:
    """Compute V_new = V_domain \\ V_base — tokens in domain BPE absent from base.

    Args:
        domain_bpe: trained domain BPE tokenizer
        base_tokenizer: HuggingFace base tokenizer (e.g. SciRus-tiny)
    Returns:
        list of new token strings to add
    """
    domain_vocab = set(domain_bpe.get_vocab().keys())
    base_vocab = set(base_tokenizer.get_vocab().keys())
    new_tokens = sorted(domain_vocab - base_vocab)
    logger.info(f"New tokens: {len(new_tokens)} (domain: {len(domain_vocab)}, base: {len(base_vocab)})")
    return new_tokens


def build_extended_tokenizer(
    base_name: str,
    new_tokens: list[str],
    cache_dir: Path | None = None,
) -> AutoTokenizer:
    """Extend base tokenizer with new domain tokens.

    Args:
        base_name: HuggingFace model name for base tokenizer
        new_tokens: list of new token strings
        cache_dir: if provided, save extended tokenizer here
    Returns:
        extended AutoTokenizer (V_base ∪ V_new)
    """
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    n_added = tokenizer.add_tokens(new_tokens)
    logger.info(f"Extended tokenizer: added {n_added} tokens, total vocab: {len(tokenizer)}")

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(cache_dir))
        logger.info(f"Saved extended tokenizer to {cache_dir}")

    return tokenizer


def fvt_initialize(
    encoder: torch.nn.Module,
    base_tokenizer: AutoTokenizer,
    extended_tokenizer: AutoTokenizer,
    new_tokens: list[str],
) -> None:
    """FVT initialization for new tokens (MATH.md M.2.3).

    For each new token t_new, compute its subword decomposition using the base
    tokenizer, then set: e(t_new) = mean(e(sub_1), ..., e(sub_k)).

    Args:
        encoder: TextEncoder (must have backbone with resized embeddings)
        base_tokenizer: original tokenizer (before extension)
        extended_tokenizer: extended tokenizer (with new tokens)
        new_tokens: list of new token strings
    """
    # Get embedding layer
    if hasattr(encoder, 'backbone'):
        model = encoder.backbone
    else:
        model = encoder

    # Find embedding weight tensor (handles BERT, ModernBERT, GPT-2, etc.)
    embed_weight = None
    if hasattr(model, 'embeddings'):
        emb = model.embeddings
        if hasattr(emb, 'word_embeddings'):
            embed_weight = emb.word_embeddings.weight
        elif hasattr(emb, 'tok_embeddings'):
            embed_weight = emb.tok_embeddings.weight  # ModernBERT
    if embed_weight is None and hasattr(model, 'embed_tokens'):
        embed_weight = model.embed_tokens.weight
    if embed_weight is None:
        logger.warning("Cannot find embedding layer for FVT init, skipping")
        return

    base_vocab_size = len(base_tokenizer)
    initialized = 0

    with torch.no_grad():
        for token_str in new_tokens:
            # Get the ID of this token in the extended tokenizer
            token_id = extended_tokenizer.convert_tokens_to_ids(token_str)
            if token_id is None or token_id < base_vocab_size:
                continue  # skip if not a new token or unknown

            # Subword decomposition using base tokenizer
            sub_ids = base_tokenizer.encode(token_str, add_special_tokens=False)
            if not sub_ids:
                continue

            # Filter valid sub_ids (must be in base vocab range)
            valid_ids = [sid for sid in sub_ids if sid < base_vocab_size]
            if not valid_ids:
                continue

            # FVT: e_new = mean(e_sub_1, ..., e_sub_k)
            sub_embeds = embed_weight[valid_ids]  # [k, d]
            embed_weight[token_id] = sub_embeds.mean(dim=0)
            initialized += 1

    logger.info(f"FVT initialized {initialized}/{len(new_tokens)} new token embeddings")


def prepare_tokenizers(
    data_dir: str | Path,
    base_name: str,
    lean_vocab_size: int = 16000,
    latex_vocab_size: int = 16000,
    cache_dir: str | Path = "tokenizers",
    max_corpus_samples: int = 0,
) -> dict:
    """Main entry point: train/load domain tokenizers, extend base, return per-modality dict.

    Args:
        data_dir: path to scilibrumodal-v2-data
        base_name: HuggingFace base tokenizer name
        lean_vocab_size: target vocab size for Lean BPE
        latex_vocab_size: target vocab size for LaTeX BPE
        cache_dir: directory to cache trained tokenizers
        max_corpus_samples: limit corpus size (0 = all)
    Returns:
        dict with keys: 'en', 'ru', 'lean', 'latex', 'base',
                        'lean_new_tokens', 'latex_new_tokens'
    """
    data_dir = Path(data_dir)
    cache_dir = Path(cache_dir)

    base_tokenizer = AutoTokenizer.from_pretrained(base_name)

    # --- Lean tokenizer ---
    lean_cache = cache_dir / "lean_extended"
    if lean_cache.exists() and (lean_cache / "tokenizer_config.json").exists():
        logger.info(f"Loading cached Lean tokenizer from {lean_cache}")
        lean_tokenizer = AutoTokenizer.from_pretrained(str(lean_cache))
        # Recover new_tokens list from vocab diff
        lean_new_tokens = sorted(set(lean_tokenizer.get_vocab().keys()) - set(base_tokenizer.get_vocab().keys()))
    else:
        logger.info("Training Lean BPE tokenizer...")
        lean_corpus = extract_domain_corpus(data_dir, "formal_statement", max_corpus_samples)
        lean_bpe = train_domain_bpe(lean_corpus, vocab_size=lean_vocab_size)
        lean_new_tokens = get_new_tokens(lean_bpe, base_tokenizer)
        lean_tokenizer = build_extended_tokenizer(base_name, lean_new_tokens, lean_cache)

    # --- LaTeX tokenizer ---
    latex_cache = cache_dir / "latex_extended"
    if latex_cache.exists() and (latex_cache / "tokenizer_config.json").exists():
        logger.info(f"Loading cached LaTeX tokenizer from {latex_cache}")
        latex_tokenizer = AutoTokenizer.from_pretrained(str(latex_cache))
        latex_new_tokens = sorted(set(latex_tokenizer.get_vocab().keys()) - set(base_tokenizer.get_vocab().keys()))
    else:
        logger.info("Training LaTeX BPE tokenizer...")
        latex_corpus = extract_domain_corpus(data_dir, "formula_md", max_corpus_samples)
        latex_bpe = train_domain_bpe(latex_corpus, vocab_size=latex_vocab_size)
        latex_new_tokens = get_new_tokens(latex_bpe, base_tokenizer)
        latex_tokenizer = build_extended_tokenizer(base_name, latex_new_tokens, latex_cache)

    logger.info(
        f"Tokenizers ready: base={len(base_tokenizer)}, "
        f"lean={len(lean_tokenizer)} (+{len(lean_new_tokens)}), "
        f"latex={len(latex_tokenizer)} (+{len(latex_new_tokens)})"
    )

    return {
        "base": base_tokenizer,
        "en": base_tokenizer,
        "ru": base_tokenizer,
        "lean": lean_tokenizer,
        "latex": latex_tokenizer,
        "lean_new_tokens": lean_new_tokens,
        "latex_new_tokens": latex_new_tokens,
    }
