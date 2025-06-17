#!/usr/bin/env python3
import functools
import math
import os
from typing import Any
import ipdb
import numpy as np
import yaml
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EvalPrediction,
)
from datasets import DatasetDict, load_dataset
import logging
import sys
import argparse
import torch
import torch.nn as nn


WHITE = "▁"
logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)


# From https://discuss.huggingface.co/t/creating-a-custom-token-vocabulary-for-gpt-2/134522
class TokenizerWrapper(PreTrainedTokenizer):  # TODO _ in decode
    def __init__(
        self,
        vocab: str | dict[str, int],
        max_len: int = 256,
        bos: str = "<s>",
        eos: str = "</s>",
        pad: str | None = None,
        additional_vocab: str | dict[str, int] | None = None,
    ):

        if isinstance(vocab, str):
            with open(vocab) as f:
                vocab = yaml.load(f, yaml.Loader)
        if additional_vocab is not None:
            if isinstance(additional_vocab, str):
                with open(additional_vocab) as f:
                    additional_vocab = yaml.load(f, yaml.Loader)
            old_vocab = sorted(vocab.items(), key=lambda x: x[1])
            old_vocab = [i[0] for i in old_vocab]
            new_vocab = {}
            for i in old_vocab:
                new_vocab[i] = len(new_vocab)
            for i in additional_vocab.keys():
                new_vocab[i] = new_vocab.get(i, len(new_vocab))
            vocab = new_vocab
            
        self.vocab = vocab
        self.id_to_token = {i: token for token, i in vocab.items()}
        self.token_to_id = vocab
        self.max_len = max_len
        self.bos_token = bos
        self.eos_token = eos
        self.pad_token = pad if pad is not None else eos
        super().__init__()

    def get_vocab(self):
        return self.vocab

    def _tokenize(self, text: str, **kargs) -> list[str]:
        return text.split()

    def _convert_token_to_id(self, token):
        return self.token_to_id.get(token, self.token_to_id.get("<unk>", 0))
    
    def _convert_id_to_token(self, index: int) -> str:
        return "" if index == -100 else self.id_to_token.get(index, "<unk>")

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens).translate({ord(WHITE): " "})
    
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
    
    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str]:
        return tuple()

def init_model(
    vocab: str, n_positions=512, bos="<s>", eos="</s>", additional_vocab: str | None = None
) -> tuple[GPT2LMHeadModel, TokenizerWrapper, GPT2Config]:
    # </s>: 2
    # <pad>: 3
    # <s>: 1
    # <unk>: 0
    tokenizer = TokenizerWrapper(vocab, additional_vocab=additional_vocab)    
    config = GPT2Config(
        vocab_size=len(tokenizer.vocab),
        n_positions=n_positions,
        bos_token_id=tokenizer.vocab[bos],
        eos_token_id=tokenizer.vocab[eos],
    )

    model = GPT2LMHeadModel(config)
    model_size = sum(t.numel() for t in model.parameters())
    logging.info(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")
    return model, tokenizer, config


def load_data(
    train: str | list[str], valid: str, test: str, tokenizer: TokenizerWrapper | None = None, max_samples=0
) -> DatasetDict:
    # ds["train"]["text"]
    ds = load_dataset("text", data_files={"train": train, "valid": valid, "test": test})
    ds = ds.shuffle(seed=42)
    sets = 1 if isinstance(train, str) else len(train)
    if max_samples > 0 and max_samples < len(ds["train"]):
        ds["train"] = ds["train"].take(max_samples * sets)
    
    if tokenizer is not None:

        def tokenize(item):
            ids = tokenizer(
                item["text"],
                truncation=True,
                max_length=tokenizer.max_len,
                add_special_tokens=True,
            )
            return ids

        ds = ds.map(tokenize, batched=True)
    return ds

# From https://github.com/huggingface/transformers/blob/main/src/transformers/loss/loss_utils.py
def fixed_cross_entropy(
    source: torch.Tensor,
    target: torch.Tensor,
    num_items_in_batch: int | None = None,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:    
    reduction = "sum" if num_items_in_batch is not None else "mean" # TODO check reduction
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss

# From https://github.com/huggingface/transformers/blob/main/src/transformers/loss/loss_utils.py
def ForCausalLMLoss(
    outputs,
    labels,
    vocab_size: int,
    num_items_in_batch: int | None = None,
    ignore_index: int = -100,
    shift_labels: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = outputs["logits"]
    logits = logits.float()

    if shift_labels is None:
        # Shift so that tokens < n predict n
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(logits.device)
    loss = fixed_cross_entropy(logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
    return loss

class Metrics:
    def __init__(self, tokenizer: TokenizerWrapper):
        self.tokenizer = tokenizer

        # logprobs = [i.logprobs for i in items]
        # weights = [i.weights for i in items]

        # if self.metric_type == "perplexity":
        #     return math.exp(-np.mean(logprobs))
        # if self.metric_type == "weighted_perplexity":
        #     return math.exp(-sum(logprobs) / sum(weights))
        # if self.metric_type == "bits_per_byte":
        #     return -sum(logprobs) / sum(weights) * 1 / math.log(2)

    def compute_metrics(self, predictions: EvalPrediction) -> dict[str, float]:
        # From https://www.skeptric.com/perplexity and https://github.com/huggingface/lighteval/blob/main/src/lighteval/metrics/metrics_corpus.py#L145
        # predictions.predictions # (samples, len?, vocab)
        # predictions.elements # tuple (samples, len?, vocab), (samples, len?), (samples, )
        # predictions.inputs # None
        # predictions.label_ids # (samples, len?)
        # predictions.losses # (samples, )
        samples = len(predictions.losses)
        cross_entropy = np.sum(predictions.losses) # (samples,), each item is the avg cross entropy
        texts = self.tokenizer.batch_decode(predictions.label_ids, skip_special_tokens=True) # list (samples)
        # tokens = (predictions.label_ids != -100).sum(1)
        tokens = (predictions.label_ids != -100).sum()
        words = np.array([len(text.split()) for text in texts]).sum()
        bytes = np.array([len(text.encode()) for text in texts]).sum()
        chars = np.array([len(text) for text in texts]).sum()
        metrics = {
            "ppl": np.exp(cross_entropy / samples),
            "ppl_token": np.exp(cross_entropy / tokens),
            "ppl_words": np.exp(cross_entropy / words),
            "ppl_chars": np.exp(cross_entropy / chars),
            "ppl_bytes": np.exp(cross_entropy / bytes),
            "bpb": cross_entropy / bytes / np.log(2), # TODO check 
            "bpc": cross_entropy / chars / np.log(2),
        }
        return metrics
    
class SaveExamplesTrainer(Trainer):
    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        ret = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        
        step = self.state.global_step
        output_dir = self.args.output_dir
        name = os.path.join(output_dir, f"validation_examples_{step}.txt")
        self.save_examples(self.eval_dataset, name)
        
        return ret
    
    def save_examples(self, eval_dataset, path, examples=3, length=10):
        dl = self.get_test_dataloader(eval_dataset)
        batch = next(iter(dl))
        input_ids = batch["input_ids"][:examples,:length]
        attention_mask = batch["attention_mask"][:examples,:length]
        with torch.no_grad():
            output_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=10)
        examples = [self.processing_class.convert_ids_to_tokens(i) for i in output_ids]

        with open(path, "w") as fout:
            for i in examples:
                fout.write(" ".join(i) + "\n")


def main(args):
    logger.info("Loading model, tokenizer and config...")
    model, tokenizer, config = init_model(args.vocab, args.max_len, additional_vocab=args.additional_vocab)    
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    logger.info("Loading data...")
    ds = load_data(
        args.train if args.train_other is None else [args.train, args.train_other], 
        args.valid, 
        args.test, 
        tokenizer, 
        args.steps
    )

    logger.info("Loading metrics...")
    metric = Metrics(tokenizer)
    loss_fn = functools.partial(ForCausalLMLoss, vocab_size=model.config.vocab_size)

    logger.info("Setting up trainer...")
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        eval_strategy="steps",
        eval_steps=math.ceil(args.eval_steps / args.batch),
        #eval_accumulation_steps=8,
        logging_steps=100,
        gradient_accumulation_steps=8,
        max_steps=math.ceil(args.steps / args.batch),
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-5,
        save_steps=math.ceil(args.eval_steps / args.batch),
        save_strategy="steps",
        save_total_limit=1,
        save_only_model=True,
        fp16=True,
        eval_on_start=True,
        # label_smoothing_factor=0.1,
        report_to=["tensorboard"],
        include_for_metrics=["loss"],
        seed=args.seed,
    )

    trainer = SaveExamplesTrainer(
        model=model,
        processing_class=tokenizer,
        args=train_args,
        data_collator=data_collator,
        train_dataset=ds["train"],
        eval_dataset=ds["valid"],
        compute_loss_func=loss_fn,
        compute_metrics=metric.compute_metrics,
        preprocess_logits_for_metrics=lambda logits, labels: logits.argmax(dim=-1)
    )
    logger.info("Start training...")
    train_results = trainer.train()
    trainer.log_metrics("train", train_results.metrics)
    
    logs = trainer.state.log_history
    train_logs = [logs[i] for i in range(1, len(logs), 2)]
    valid_logs = [logs[i] for i in range(0, len(logs), 2)]

    logger.info("Testing...")
    test_metrics = trainer.evaluate(ds["test"])
    trainer.log_metrics("test", test_metrics)

    trainer.save_model(args.output_dir)
    trainer.save_metrics("all", {"train": train_logs, "valid": valid_logs, "test": test_metrics})        
    
    trainer.save_examples(ds["test"], os.path.join(args.output_dir, "test_examples.txt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", type=str, required=True, help="Train data (tokenized)."
    )
    parser.add_argument(
        "--train-other", type=str, required=False, default=None, help="Train data (tokenized)."
    )
    parser.add_argument(
        "--valid", type=str, required=True, help="Validation data (tokenized)."
    )
    parser.add_argument(
        "--test", type=str, required=True, help="Test data (tokenized)."
    )
    parser.add_argument(
        "--vocab", type=str, required=True, help="Vocabulary file (yaml)."
    )
    parser.add_argument(
        "--additional-vocab", type=str, default=None, help="Vocabulary file (yaml)."
    )
    parser.add_argument(
        "--max-len", type=int, default=512, help="Maximum sequence length."
    )

    parser.add_argument(
        "--batch", type=int, default=128, help="Training batch size."
    )

    parser.add_argument(
        "--steps", type=int, default=4_000_000, help="Training examples."
    )

    parser.add_argument(
        "--eval-steps", type=int, default=200_000, help="Validation examples."
    )

    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=5)

    args = parser.parse_args()
    main(args)
