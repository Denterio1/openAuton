"""
Real Training Simulator

This module provides a lightweight trainer for actual model training using PyTorch.
It supports:
- Loading a small transformer model (GPT‑style) from HuggingFace
- Training on synthetic data
- Saving checkpoints
- Returning evaluation metrics

This allows the agent to actually train models instead of just simulating.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset as HFDataset
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import json
import random

from src.training.plan import ArchitectureConfig, DataConfig, TrainingConfig, EvaluationMetrics, HallucinationLevel

logger = logging.getLogger(__name__)


class SyntheticTextDataset(Dataset):
    """Simple synthetic dataset for training."""
    
    def __init__(self, tokenizer, num_samples: int = 1000, max_length: int = 128):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_length = max_length
        
        # Generate synthetic text patterns
        self.texts = []
        for i in range(num_samples):
            # Create varied text patterns
            pattern = random.choice([
                "The answer to question {i} is {ans}.",
                "Given the context, we can conclude {ans}.",
                "Step by step reasoning: first, consider {i}. Then, {ans}.",
                "Analysis shows that {ans} is correct for input {i}."
            ])
            ans = random.choice(["yes", "no", "42", "the result is positive"])
            text = pattern.format(i=i, ans=ans)
            self.texts.append(text)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length, return_tensors="pt")
        return enc.input_ids.squeeze(0)


class ModelTrainer:
    """
    Real model trainer using HuggingFace Transformers.
    Handles training, evaluation, and metric collection.
    """
    
    def __init__(self, output_dir: Path = Path("experiments/models")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def train(self,
              arch_config: ArchitectureConfig,
              data_config: DataConfig,
              training_config: TrainingConfig,
              objective: str = "next_token_prediction") -> EvaluationMetrics:
        """
        Train a model based on the provided configurations.
        
        Returns evaluation metrics after training.
        """
        logger.info("Starting real training...")
        
        # 1. Choose model based on architecture config
        model_name = self._select_model(arch_config)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 2. Load model
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # 3. Prepare dataset
        dataset = self._prepare_dataset(tokenizer, data_config)
        
        # 4. Set up training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "checkpoints"),
            overwrite_output_dir=True,
            num_train_epochs=training_config.max_steps // len(dataset) if training_config.max_steps else 3,
            per_device_train_batch_size=training_config.batch_size,
            per_device_eval_batch_size=training_config.batch_size,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            learning_rate=training_config.learning_rate,
            warmup_steps=training_config.warmup_steps,
            weight_decay=training_config.weight_decay,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=training_config.eval_every,
            save_steps=training_config.save_every,
            save_total_limit=2,
            load_best_model_at_end=True,
            fp16=training_config.use_fp16,
            bf16=training_config.use_bf16,
            dataloader_drop_last=False,
        )
        
        # 5. Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # 6. Create Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset,  # for demo, using same
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        
        # 7. Train
        trainer.train()
        
        # 8. Evaluate
        eval_results = trainer.evaluate()
        
        # 9. Compute metrics
        metrics = self._compute_metrics(eval_results, model, tokenizer, dataset)
        
        # 10. Save final model
        trainer.save_model(str(self.output_dir / "final_model"))
        tokenizer.save_pretrained(str(self.output_dir / "final_model"))
        
        logger.info("Training completed.")
        return metrics
    
    def _select_model(self, arch_config: ArchitectureConfig) -> str:
        """Map architecture config to a HuggingFace model name."""
        # For simplicity, use a small model.
        # In real implementation, you'd adjust based on arch_config.
        return "gpt2"  # 124M parameters, good for testing
    
    def _prepare_dataset(self, tokenizer, data_config: DataConfig) -> HFDataset:
        """Create a synthetic dataset of the specified size."""
        # In production, you would load real data.
        # For demo, generate synthetic text.
        texts = []
        for i in range(min(data_config.dataset_size, 10000)):  # limit for speed
            # Simple synthetic pattern
            text = f"This is sample number {i}. " * 10
            texts.append(text)
        
        # Tokenize
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, max_length=512)
        
        hf_dataset = HFDataset.from_dict({"text": texts})
        tokenized_dataset = hf_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        return tokenized_dataset
    
    def _compute_metrics(self,
                         eval_results: Dict[str, float],
                         model,
                         tokenizer,
                         dataset) -> EvaluationMetrics:
        """Compute evaluation metrics from training results."""
        # Basic metrics
        val_loss = eval_results.get("eval_loss", 2.0)
        perplexity = np.exp(val_loss)
        
        # Dummy reasoning score (in real, evaluate on reasoning tasks)
        reasoning_score = 0.5 + (1.0 - min(1.0, val_loss / 5.0)) * 0.5
        
        # Simple hallucination estimate: high perplexity may indicate hallucinations
        hallucination_level = HallucinationLevel.MODERATE if perplexity > 50 else HallucinationLevel.MINOR
        
        # Dummy bias and safety
        bias_score = 0.2
        safety_score = 0.8
        
        return EvaluationMetrics(
            train_loss=eval_results.get("train_loss", val_loss),
            val_loss=val_loss,
            perplexity=perplexity,
            accuracy=0.7,  # dummy
            reasoning_score=reasoning_score,
            hallucination_level=hallucination_level,
            bias_score=bias_score,
            safety_score=safety_score,
            training_time_hours=0.5,  # dummy
            gpu_hours=0.5,
            flops_utilization=0.6
        )