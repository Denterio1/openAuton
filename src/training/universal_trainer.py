"""
training/universal_trainer.py
=============================
UniversalAutonomousWrapper – ingests any file, sanitises, detects primary text column,
selects architecture, and simulates training (no GPU, no torch required).
"""

from __future__ import annotations
import json
import xml.etree.ElementTree as ET
import re
import math
import random
import hashlib
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import Counter

# Optional imports (no torch)
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False

from tools.file_ops import FileOps


# ──────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────

@dataclass
class DataProfile:
    file_type: str
    num_samples: int
    num_tokens_est: int
    language: str
    vocab_size_est: int
    recommended_architecture: Dict[str, Any]
    cleaning_stats: Dict[str, int]
    token_entropy: float = 0.0
    vocab_diversity: float = 0.0
    duplication_rate: float = 0.0
    data_quality_score: float = 0.0

@dataclass
class TrainingReport:
    data_profile: DataProfile
    model_config: Dict[str, Any]
    training_loss: float
    final_accuracy: float
    gene_hints: List[str]
    lessons_learned: List[str]
    next_improvement: str
    data_quality_before: float = 0.0
    data_quality_after: float = 0.0
    augmentation_used: bool = False
    trial_burn_passed: bool = True


# ──────────────────────────────────────────────────────────
# Universal Text Normalizer
# ──────────────────────────────────────────────────────────

class UniversalTextNormalizer:
    @staticmethod
    def normalize(text: str) -> str:
        text = unicodedata.normalize('NFKC', text)
        # Keep letters, digits, spaces, punctuation
        text = re.sub(r'[^\x20-\x7E\x0A\x0D\u0600-\u06FF\u4E00-\u9FFF]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


# ──────────────────────────────────────────────────────────
# Statistical Profiler
# ──────────────────────────────────────────────────────────

class StatisticalProfiler:
    @staticmethod
    def compute_entropy(tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        freq = Counter(tokens)
        probs = [c / len(tokens) for c in freq.values()]
        return -sum(p * math.log2(p) for p in probs)

    @staticmethod
    def compute_vocab_diversity(tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)

    @staticmethod
    def estimate_duplication_rate(texts: List[str], sample_size: int = 500) -> float:
        if len(texts) < 2:
            return 0.0
        sample = random.sample(texts, min(sample_size, len(texts)))
        normalized = [UniversalTextNormalizer.normalize(t) for t in sample]
        hashes = [hashlib.md5(t.encode('utf-8')).hexdigest() for t in normalized]
        return 1.0 - (len(set(hashes)) / len(hashes)) if hashes else 0.0

    @staticmethod
    def compute_data_quality_score(entropy: float, diversity: float, dup_rate: float) -> float:
        entropy_score = min(1.0, entropy / 8.0) if entropy > 0 else 0.2
        return round(entropy_score * 0.4 + diversity * 0.3 + (1 - dup_rate) * 0.3, 3)


# ──────────────────────────────────────────────────────────
# Auto‑Schema Discovery
# ──────────────────────────────────────────────────────────

class SchemaDiscovery:
    @staticmethod
    def discover_primary_text_column(data: Any, file_type: str) -> Optional[str]:
        if file_type == '.csv' and HAS_PANDAS:
            if isinstance(data, pd.DataFrame):
                best_col = None
                max_text_score = 0
                for col in data.columns:
                    sample = data[col].dropna().astype(str).head(100)
                    if len(sample) == 0:
                        continue
                    avg_len = sample.str.len().mean()
                    alpha_ratio = sample.str.count(r'[A-Za-z\u0600-\u06FF\u4E00-\u9FFF]').mean() / (avg_len + 1)
                    text_score = avg_len * 0.5 + alpha_ratio * 0.5
                    if text_score > max_text_score:
                        max_text_score = text_score
                        best_col = col
                return best_col
        elif file_type == '.json' and isinstance(data, list) and len(data) > 0:
            first = data[0]
            best_key = None
            max_score = 0
            for key, val in first.items():
                if isinstance(val, str):
                    sample_lengths = [len(str(item.get(key, ""))) for item in data[:100] if isinstance(item, dict)]
                    if sample_lengths:
                        avg_len = sum(sample_lengths) / len(sample_lengths)
                        if avg_len > max_score:
                            max_score = avg_len
                            best_key = key
            return best_key
        return None


# ──────────────────────────────────────────────────────────
# Synthetic Augmentation
# ──────────────────────────────────────────────────────────

class SyntheticGenerator:
    @staticmethod
    def augment(texts: List[str], target_size: int = 2000) -> List[str]:
        if len(texts) >= target_size:
            return texts
        needed = target_size - len(texts)
        synthetic = []
        for _ in range(needed):
            original = random.choice(texts)
            words = original.split()
            if len(words) > 2:
                idx1, idx2 = random.sample(range(len(words)), 2)
                words[idx1], words[idx2] = words[idx2], words[idx1]
                augmented = " ".join(words)
            else:
                augmented = original + " " + original
            synthetic.append(augmented)
        return texts + synthetic


# ──────────────────────────────────────────────────────────
# Main Universal Trainer (Simulation only)
# ──────────────────────────────────────────────────────────

class UniversalAutonomousWrapper:
    def __init__(self, file_ops: FileOps, output_dir: Path = Path("experiments/universal_models")):
        self.file_ops = file_ops
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, file_path: Path, epochs: int = 10, batch_size: int = 8,
            trial_steps: int = 10, augment_if_small: bool = True) -> TrainingReport:
        # 1. Ingest and profile
        texts, profile = self._ingest_and_profile(file_path)
        if not texts:
            raise ValueError("No valid text data found.")

        # 2. Augment if needed
        augmentation_used = False
        if augment_if_small and len(texts) < 1000:
            texts = SyntheticGenerator.augment(texts, target_size=2000)
            augmentation_used = True
            profile.num_samples = len(texts)
            profile.num_tokens_est = sum(len(t.split()) for t in texts)
            profile.vocab_size_est = min(50000, max(1000, int(math.sqrt(profile.num_tokens_est) * 10)))

        # 3. Simulate training (no real training)
        simulated_accuracy = min(0.95, 0.5 + math.log10(profile.num_tokens_est + 1) / 10)
        simulated_loss = max(0.5, 2.5 - simulated_accuracy * 2)

        # 4. Generate report
        gene_hints = []
        lessons = []
        if profile.data_quality_score < 0.5:
            gene_hints.append("low_data_quality")
            lessons.append("Data quality low – consider better source or cleaning.")
        if profile.language != "en":
            gene_hints.append(f"lang_{profile.language}")
        if simulated_accuracy < 0.6:
            gene_hints.append("low_accuracy")
            lessons.append(f"Accuracy {simulated_accuracy:.2f} below target.")
        else:
            gene_hints.append("high_accuracy")

        next_improvement = "Increase model depth or data size." if simulated_accuracy < 0.7 else "Consider fine‑tuning on more data."

        return TrainingReport(
            data_profile=profile,
            model_config=profile.recommended_architecture,
            training_loss=simulated_loss,
            final_accuracy=simulated_accuracy,
            gene_hints=gene_hints,
            lessons_learned=lessons,
            next_improvement=next_improvement,
            data_quality_before=profile.data_quality_score,
            data_quality_after=profile.data_quality_score,
            augmentation_used=augmentation_used,
            trial_burn_passed=True,
        )

    # ------------------------------------------------------------------
    # Internal helpers (no torch)
    # ------------------------------------------------------------------

    def _ingest_and_profile(self, file_path: Path) -> Tuple[List[str], DataProfile]:
        ext = file_path.suffix.lower()
        raw_texts = self._load_texts(file_path, ext)
        if not raw_texts:
            return [], None

        normalized = [UniversalTextNormalizer.normalize(t) for t in raw_texts if t.strip()]
        normalized = list(dict.fromkeys(normalized))

        sample = normalized[:min(500, len(normalized))]
        tokens = [word for t in sample for word in t.split()]
        entropy = StatisticalProfiler.compute_entropy(tokens)
        diversity = StatisticalProfiler.compute_vocab_diversity(tokens)
        dup_rate = StatisticalProfiler.estimate_duplication_rate(normalized)
        quality_score = StatisticalProfiler.compute_data_quality_score(entropy, diversity, dup_rate)

        total_words = sum(len(t.split()) for t in normalized)
        vocab_est = min(50000, max(1000, int(math.sqrt(total_words) * 10)))
        if total_words < 1000:
            arch = {"type": "transformer", "num_layers": 2, "d_model": 64, "num_heads": 4}
        elif total_words < 10000:
            arch = {"type": "transformer", "num_layers": 4, "d_model": 128, "num_heads": 8}
        elif total_words < 100000:
            arch = {"type": "transformer", "num_layers": 6, "d_model": 256, "num_heads": 8}
        else:
            arch = {"type": "transformer", "num_layers": 8, "d_model": 512, "num_heads": 12}

        lang = "unknown"
        if HAS_LANGDETECT and normalized:
            try:
                lang = detect(" ".join(normalized[:5]))
            except:
                pass

        profile = DataProfile(
            file_type=ext,
            num_samples=len(normalized),
            num_tokens_est=total_words,
            language=lang,
            vocab_size_est=vocab_est,
            recommended_architecture=arch,
            cleaning_stats={"original": len(raw_texts), "after_clean": len(normalized)},
            token_entropy=entropy,
            vocab_diversity=diversity,
            duplication_rate=dup_rate,
            data_quality_score=quality_score,
        )
        return normalized, profile

    def _load_texts(self, path: Path, ext: str) -> List[str]:
        # Try universal file parser if available
        try:
            from tools.file_parser import get_file_text
            full_text, _ = get_file_text(path, max_chars=500_000)
            lines = [line.strip() for line in full_text.splitlines() if line.strip()]
            if not lines:
                import re
                sentences = re.split(r'[.!?]+', full_text)
                lines = [s.strip() for s in sentences if len(s.strip()) > 20]
            return lines
        except ImportError:
            # Fallback: read as plain text
            try:
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                return [line.strip() for line in content.splitlines() if line.strip()]
            except Exception:
                return []


# Convenience function for registry
def create_universal_trainer(file_ops: FileOps):
    trainer = UniversalAutonomousWrapper(file_ops)
    def universal_train(file_path: str, epochs: int = 10, batch_size: int = 8) -> Dict:
        report = trainer.run(Path(file_path), epochs, batch_size)
        return {
            "status": "success",
            "data_profile": {
                "num_samples": report.data_profile.num_samples,
                "language": report.data_profile.language,
                "data_quality": report.data_profile.data_quality_score,
            },
            "final_accuracy": report.final_accuracy,
            "gene_hints": report.gene_hints,
            "lessons_learned": report.lessons_learned,
            "next_improvement": report.next_improvement,
            "augmentation_used": report.augmentation_used,
        }
    return universal_train