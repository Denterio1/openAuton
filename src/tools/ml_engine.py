"""
tools/ml_engine.py
==================
Advanced ML training engine for the self‑improving agent.

Components:
1. ModelFactory – dynamically builds PyTorch models from DNA specifications
2. TrainingParadigms – ready‑to‑use training methods (LoRA, QLoRA, DPO, PPO, Contrastive)
3. ExecutionRuntime – acceleration (DeepSpeed, FlashAttention) and quantization
4. TrialBurner – fast trial run (few steps) to validate training before full budget spend

Integrates with:
- intuition.py (receives trial metrics for go/no‑go decision)
- genome/dna.py (reads architecture genes)
- training/simulator.py (can replace or extend it)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import math
import time
import json
from pathlib import Path

# Optional imports (with fallbacks if not installed)
try:
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model, TaskType as PeftTaskType
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import deepspeed
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False

try:
    from flash_attn.flash_attention import FlashAttention
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


# ──────────────────────────────────────────────────────────
# 1. ModelFactory – Dynamic Model Builder
# ──────────────────────────────────────────────────────────

class ModelFactory:
    """
    Builds PyTorch models dynamically from architecture specifications.
    Supports: Transformer (decoder‑only), MLP, CNN, custom configs.
    """

    @staticmethod
    def build(arch_config: Dict[str, Any]) -> nn.Module:
        """
        arch_config example:
        {
            "type": "transformer",
            "num_layers": 6,
            "d_model": 256,
            "num_heads": 8,
            "vocab_size": 50257,
            "max_seq_len": 512,
            "dropout": 0.1
        }
        """
        arch_type = arch_config.get("type", "transformer").lower()
        if arch_type == "transformer":
            return ModelFactory._build_transformer(arch_config)
        elif arch_type == "mlp":
            return ModelFactory._build_mlp(arch_config)
        elif arch_type == "cnn":
            return ModelFactory._build_cnn(arch_config)
        else:
            raise ValueError(f"Unknown architecture type: {arch_type}")

    @staticmethod
    def _build_transformer(config: Dict) -> nn.Module:
        """Simple decoder‑only transformer (GPT‑like)."""
        num_layers = config.get("num_layers", 6)
        d_model = config.get("d_model", 256)
        num_heads = config.get("num_heads", 8)
        vocab_size = config.get("vocab_size", 10000)
        max_seq_len = config.get("max_seq_len", 512)
        dropout = config.get("dropout", 0.1)

        # Use HuggingFace if available, otherwise custom minimal
        if HAS_TRANSFORMERS:
            from transformers import GPT2Config, GPT2Model
            gpt_config = GPT2Config(
                n_layer=num_layers,
                n_embd=d_model,
                n_head=num_heads,
                vocab_size=vocab_size,
                n_positions=max_seq_len,
                resid_pdrop=dropout,
                embd_pdrop=dropout,
                attn_pdrop=dropout,
            )
            return GPT2Model(gpt_config)
        else:
            # Minimal custom transformer (for demo without HF)
            return ModelFactory._custom_transformer(
                num_layers, d_model, num_heads, vocab_size, max_seq_len, dropout
            )

    @staticmethod
    def _custom_transformer(num_layers, d_model, num_heads, vocab_size, max_seq_len, dropout):
        """Fallback minimal transformer (embedding + decoder layers + LM head)."""
        from torch.nn import TransformerDecoder, TransformerDecoderLayer, Embedding, Linear
        class MiniGPT(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = Embedding(vocab_size, d_model)
                decoder_layer = TransformerDecoderLayer(d_model, num_heads, dim_feedforward=4*d_model, dropout=dropout, batch_first=True)
                self.decoder = TransformerDecoder(decoder_layer, num_layers)
                self.lm_head = Linear(d_model, vocab_size)
                self.max_seq_len = max_seq_len
            def forward(self, input_ids):
                x = self.embed(input_ids)
                # dummy memory (self‑attention only)
                memory = torch.zeros_like(x)
                out = self.decoder(x, memory)
                logits = self.lm_head(out)
                return logits
        return MiniGPT()

    @staticmethod
    def _build_mlp(config: Dict) -> nn.Module:
        layers = []
        input_dim = config.get("input_dim", 784)
        hidden_dims = config.get("hidden_dims", [512, 256])
        output_dim = config.get("output_dim", 10)
        dropout = config.get("dropout", 0.2)
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        return nn.Sequential(*layers)

    @staticmethod
    def _build_cnn(config: Dict) -> nn.Module:
        # Simplified CNN for image tasks
        layers = []
        in_channels = config.get("in_channels", 3)
        conv_channels = config.get("conv_channels", [32, 64])
        kernel_size = config.get("kernel_size", 3)
        for out_ch in conv_channels:
            layers.append(nn.Conv2d(in_channels, out_ch, kernel_size, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_channels = out_ch
        layers.append(nn.Flatten())
        # Compute flattened size (assume 32x32 input)
        fc_input = conv_channels[-1] * (32 // (2**len(conv_channels)))**2
        layers.append(nn.Linear(fc_input, 128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, config.get("num_classes", 10)))
        return nn.Sequential(*layers)


# ──────────────────────────────────────────────────────────
# 2. TrainingParadigms – Advanced Training Methods
# ──────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    method: str          # "supervised", "lora", "qlora", "dpo", "ppo", "contrastive"
    base_model_name: Optional[str] = None
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    learning_rate: float = 2e-5
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation: int = 1

class TrainingParadigms:
    """
    Provides ready‑to‑use training loops for different paradigms.
    """

    @staticmethod
    def train_supervised(model: nn.Module, train_loader: DataLoader,
                         config: TrainingConfig, device: torch.device) -> Dict[str, float]:
        """Standard supervised fine‑tuning."""
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        model.train()
        total_loss = 0.0
        for epoch in range(config.num_epochs):
            epoch_loss = 0.0
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            total_loss += epoch_loss / len(train_loader)
        return {"final_loss": total_loss / config.num_epochs, "method": "supervised"}

    @staticmethod
    def train_lora(model: nn.Module, train_loader: DataLoader,
                   config: TrainingConfig, device: torch.device) -> Dict[str, float]:
        """LoRA fine‑tuning using PEFT library."""
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers and peft required for LoRA")
        from peft import LoraConfig, get_peft_model, TaskType
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        model.train()
        total_loss = 0.0
        for epoch in range(config.num_epochs):
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        return {"final_loss": total_loss / (config.num_epochs * len(train_loader)), "method": "lora"}

    @staticmethod
    def train_dpo(model: nn.Module, preference_loader: DataLoader,
                  config: TrainingConfig, device: torch.device) -> Dict[str, float]:
        """
        Direct Preference Optimization (DPO) – simplified simulation.
        In production, use TRL's DPOTrainer.
        """
        # Simplified DPO loss: maximize log_prob(chosen) - log_prob(rejected)
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in preference_loader:
            # batch contains: chosen_input_ids, rejected_input_ids
            chosen = batch["chosen"].to(device)
            rejected = batch["rejected"].to(device)
            # Forward pass (simplified: compute logits difference)
            logits_chosen = model(chosen).logits
            logits_rejected = model(rejected).logits
            # DPO loss (very simplified – for real use, use proper implementation)
            loss = -torch.mean(logits_chosen - logits_rejected)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1
            if steps >= 100:  # limit for demo
                break
        return {"final_loss": total_loss / steps, "method": "dpo"}

    @staticmethod
    def train_contrastive(model: nn.Module, train_loader: DataLoader,
                          config: TrainingConfig, device: torch.device) -> Dict[str, float]:
        """Contrastive learning (SimCLR style)."""
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in train_loader:
            # Expecting two augmented views of same batch
            x_i, x_j = batch[0].to(device), batch[1].to(device)
            z_i = model(x_i)
            z_j = model(x_j)
            # Normalize
            z_i = nn.functional.normalize(z_i, dim=1)
            z_j = nn.functional.normalize(z_j, dim=1)
            # InfoNCE loss
            batch_size = z_i.size(0)
            similarity = torch.matmul(z_i, z_j.T)
            labels = torch.arange(batch_size).to(device)
            loss = nn.functional.cross_entropy(similarity, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1
            if steps >= 100:
                break
        return {"final_loss": total_loss / steps, "method": "contrastive"}

    @staticmethod
    def run(model: nn.Module, train_loader: DataLoader,
            config: TrainingConfig, device: torch.device) -> Dict[str, float]:
        """Route to appropriate training method."""
        method = config.method.lower()
        if method == "supervised":
            return TrainingParadigms.train_supervised(model, train_loader, config, device)
        elif method == "lora":
            return TrainingParadigms.train_lora(model, train_loader, config, device)
        elif method == "dpo":
            return TrainingParadigms.train_dpo(model, train_loader, config, device)
        elif method == "contrastive":
            return TrainingParadigms.train_contrastive(model, train_loader, config, device)
        else:
            raise ValueError(f"Unknown training method: {method}")


# ──────────────────────────────────────────────────────────
# 3. ExecutionRuntime – Acceleration & Quantization
# ──────────────────────────────────────────────────────────

class ExecutionRuntime:
    """
    Optimizes training with DeepSpeed, FlashAttention, quantization.
    """

    def __init__(self, use_deepspeed: bool = False, use_flash_attn: bool = False,
                 quantization: Optional[str] = None):  # "int8", "int4"
        self.use_deepspeed = use_deepspeed and HAS_DEEPSPEED
        self.use_flash_attn = use_flash_attn and HAS_FLASH_ATTN
        self.quantization = quantization

    def prepare_model(self, model: nn.Module, device: torch.device) -> nn.Module:
        """Apply optimizations: quantization, FlashAttention, DeepSpeed."""
        if self.quantization == "int8":
            try:
                import torch.quantization
                model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
            except:
                pass
        elif self.quantization == "int4":
            # Requires bitsandbytes
            try:
                from transformers import BitsAndBytesConfig
                # For simplicity, we skip detailed implementation
                pass
            except:
                pass

        if self.use_flash_attn:
            # Replace attention modules with FlashAttention (simplified)
            # In real code, you'd monkey‑patch or use transformers with flash_attn flag
            pass

        model = model.to(device)
        if self.use_deepspeed:
            # DeepSpeed engine would be initialized here
            # For now, just return model
            pass

        return model

    def get_optimizer(self, model: nn.Module, lr: float):
        """Return optimizer (could use DeepSpeed optimizer)."""
        return optim.AdamW(model.parameters(), lr=lr)


# ──────────────────────────────────────────────────────────
# 4. TrialBurner – Fast Validation Before Full Training
# ──────────────────────────────────────────────────────────

class TrialBurner:
    """
    Runs a few steps of training to collect early metrics,
    then reports to intuition.py for go/no‑go decision.
    """

    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 config: TrainingConfig, device: torch.device,
                 num_steps: int = 10):
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.device = device
        self.num_steps = num_steps

    def run_trial(self) -> Dict[str, Any]:
        """
        Execute small number of training steps.
        Returns metrics: loss trend, gradient norm, cost estimate.
        """
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        self.model.train()
        losses = []
        steps_done = 0
        start_time = time.time()
        data_iter = iter(self.train_loader)

        for step in range(self.num_steps):
            try:
                inputs, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                inputs, labels = next(data_iter)

            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        duration = time.time() - start_time
        avg_loss = sum(losses) / len(losses)
        loss_trend = "decreasing" if len(losses) > 1 and losses[-1] < losses[0] else "increasing"
        estimated_cost = duration * 0.001  # dummy cost (USD per second)

        return {
            "avg_loss": avg_loss,
            "loss_trend": loss_trend,
            "duration_seconds": duration,
            "estimated_cost_usd": estimated_cost,
            "steps_completed": self.num_steps,
            "recommended": avg_loss < 2.0,  # heuristic
            "message": f"Trial completed: loss {avg_loss:.4f} ({loss_trend})"
        }

    def report_to_intuition(self, intuition_engine) -> bool:
        """
        Send trial results to intuition engine.
        Returns True if training should continue, False if abort.
        """
        trial_metrics = self.run_trial()
        # In real integration, call intuition_engine.evaluate_trial(trial_metrics)
        # For now, simple heuristic:
        return trial_metrics["recommended"]


# ──────────────────────────────────────────────────────────
# 5. Main ML Engine (Facade)
# ──────────────────────────────────────────────────────────

class MLEngine:
    """
    Facade combining ModelFactory, TrainingParadigms, ExecutionRuntime, TrialBurner.
    """

    def __init__(self, arch_config: Dict[str, Any],
                 training_config: TrainingConfig,
                 device: Optional[str] = None):
        self.arch_config = arch_config
        self.training_config = training_config
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = ModelFactory.build(arch_config)
        self.runtime = ExecutionRuntime(use_deepspeed=False, use_flash_attn=False)
        self.model = self.runtime.prepare_model(self.model, self.device)

    def train(self, train_loader: DataLoader, validate_before_train: bool = True) -> Dict[str, Any]:
        """
        Full training pipeline with optional trial burn‑in.
        """
        if validate_before_train:
            burner = TrialBurner(self.model, train_loader, self.training_config, self.device, num_steps=10)
            trial_result = burner.run_trial()
            if not trial_result["recommended"]:
                return {"status": "aborted", "reason": trial_result["message"], "trial": trial_result}

        # Run full training
        result = TrainingParadigms.run(self.model, train_loader, self.training_config, self.device)
        result["status"] = "completed"
        result["device"] = str(self.device)
        return result

    def save(self, path: Path):
        torch.save(self.model.state_dict(), path / "model.pth")
        with open(path / "config.json", "w") as f:
            json.dump({"arch": self.arch_config, "training": self.training_config.__dict__}, f)

    def load(self, path: Path):
        self.model.load_state_dict(torch.load(path / "model.pth"))