"""Training wrapper for local LoRA / QLoRA fine-tuning."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from utils.helpers import ensure_directory


BASE_MODELS = {
    "unsloth-llama-3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama-3-8b": "meta-llama/Llama-3.1-8B-Instruct",
}


@dataclass
class LocalTrainerConfig:
    base_model: str = "mistral-7b"
    dataset_path: str = ""
    learning_rate: float = 2e-4
    batch_size: int = 2
    epochs: int = 2
    lora_r: int = 16
    lora_alpha: int = 32
    max_steps: int = 0
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    output_dir: str = "outputs"


class _TrainingProgressCallback:
    """
    Small callback compatible with both Transformers and manual call sites.
    """

    def __init__(self, on_progress: Optional[Callable[[str], None]]) -> None:
        self.on_progress = on_progress

    def on_step_end(self, step: int, logs: Dict[str, Any]) -> None:
        if not self.on_progress:
            return
        loss = logs.get("loss")
        if loss is None:
            return
        msg = f"[local-trainer] step={step} loss={loss:.6f}"
        if "learning_rate" in logs:
            msg += f" lr={logs['learning_rate']:.2e}"
        self.on_progress(msg)


def _import_training_stack():
    """
    Import the full training stack lazily so the module loads without expensive deps.
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainerCallback,
    )
    from trl import SFTTrainer, SFTConfig

    return {
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "Dataset": Dataset,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "TrainerCallback": TrainerCallback,
        "SFTTrainer": SFTTrainer,
        "SFTConfig": SFTConfig,
    }


class LocalTrainer:
    """
    Train a LoRA/QLoRA adapter using SFTTrainer.
    """

    def __init__(self, output_root: str | Path = "outputs") -> None:
        self.output_root = ensure_directory(output_root)
        self.logger = logging.getLogger("CoreForge.Trainer")
        self._stack: Optional[Dict[str, Any]] = None

    def _load_stack(self):
        if self._stack is None:
            self._stack = _import_training_stack()
        return self._stack

    def _load_records(self, dataset_path: str | Path) -> List[Dict[str, str]]:
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        rows: List[Dict[str, str]] = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not all(key in payload for key in ("instruction", "input", "output")):
                    continue
                rows.append(
                    {
                        "instruction": str(payload["instruction"]),
                        "input": str(payload["input"]),
                        "output": str(payload["output"]),
                    }
                )
        if not rows:
            raise RuntimeError("Dataset file is empty or invalid.")
        return rows

    def _build_prompt(self, row: Dict[str, str], tokenizer) -> str:
        return (
            f"### Instruction:\n{row['instruction']}\n"
            f"### Input:\n{row['input']}\n"
            f"### Output:\n{row['output']}{tokenizer.eos_token or ''}"
        )

    def _build_hf_dataset(self, rows: List[Dict[str, str]], tokenizer) -> Any:
        stack = self._load_stack()
        ds_cls = stack["Dataset"]

        prompts = [self._build_prompt(item, tokenizer) for item in rows]
        encoded = {"text": prompts}
        # Using from_dict to avoid relying on column features and avoid strict schema needs.
        return ds_cls.from_dict(encoded)

    def _build_model_for_training(self, base_model: str, lora_r: int, lora_alpha: int):
        stack = self._load_stack()
        AutoModelForCausalLM = stack["AutoModelForCausalLM"]
        AutoTokenizer = stack["AutoTokenizer"]
        BitsAndBytesConfig = stack["BitsAndBytesConfig"]
        LoraConfig = stack["LoraConfig"]
        get_peft_model = stack["get_peft_model"]
        prepare_model_for_kbit_training = stack["prepare_model_for_kbit_training"]

        hf_model = BASE_MODELS.get(base_model, base_model)
        tokenizer = AutoTokenizer.from_pretrained(hf_model, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            hf_model,
            quantization_config=quantization,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        peft_config = LoraConfig(
            r=max(1, int(lora_r)),
            lora_alpha=max(1, int(lora_alpha)),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        return model, tokenizer, hf_model, peft_config

    def _train(self, cfg: LocalTrainerConfig, on_log: Optional[Callable[[str], None]] = None) -> Dict[str, str]:
        stack = self._load_stack()
        SFTTrainer = stack["SFTTrainer"]
        SFTConfig = stack["SFTConfig"]
        TrainerCallback = stack["TrainerCallback"]

        rows = self._load_records(cfg.dataset_path)
        model, tokenizer, hf_name, peft_config = self._build_model_for_training(
            cfg.base_model,
            lora_r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
        )
        hf_dataset = self._build_hf_dataset(rows, tokenizer)

        run_name = f"coreforge-{int(time.time())}"
        run_output = self.output_root / run_name
        ensure_directory(run_output)
        peft_dir = run_output / "adapter"

        class _ProgressCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if on_log and logs:
                    _TrainingProgressCallback(on_log).on_step_end(state.global_step, logs)
                return control

        trainable_config = SFTConfig(
            output_dir=str(run_output),
            dataset_text_field="text",
            max_seq_length=cfg.max_seq_length,
            learning_rate=cfg.learning_rate,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            num_train_epochs=cfg.epochs,
            logging_steps=1,
            max_steps=cfg.max_steps if cfg.max_steps > 0 else -1,
            bf16=False,
            fp16=True,
            optim="paged_adamw_8bit",
            save_strategy="no",
            report_to=["none"],
            run_name=run_name,
        )

        trainer = SFTTrainer(
            model=model,
            args=trainable_config,
            train_dataset=hf_dataset,
            tokenizer=tokenizer,
            callbacks=[_ProgressCallback()],
        )

        if on_log:
            on_log(f"Starting training run={run_name} on model={hf_name}")

        trainer.train()
        model.save_pretrained(peft_dir)
        tokenizer.save_pretrained(peft_dir)

        result = {
            "run_name": run_name,
            "run_output_dir": str(run_output.resolve()),
            "adapter_dir": str(peft_dir.resolve()),
            "base_model": hf_name,
        }
        if on_log:
            on_log(f"Training finished successfully. adapter_dir={result['adapter_dir']}")
        return result

    def train(
        self,
        config: LocalTrainerConfig,
        on_log: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, str]:
        """
        Execute training and expose callbacks for UI streaming.
        """
        try:
            return self._train(config, on_log=on_log)
        except Exception as exc:
            # Attempt to preserve existing log context for UI troubleshooting.
            msg = f"Training failed: {exc}"
            self.logger.exception("Training failed")
            if on_log:
                on_log(msg)
            raise RuntimeError(msg) from exc
