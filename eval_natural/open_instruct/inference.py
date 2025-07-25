# !/usr/bin/env python
# coding=utf-8
# Copyright 2024 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import math
import os
import random
import subprocess
import time
from dataclasses import dataclass, field
from datetime import timedelta
from functools import partial
from typing import List, Optional, Union

import datasets
import deepspeed
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import InitProcessGroupKwargs, set_seed
from datasets import load_dataset
from huggingface_hub import HfApi
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    GPT2Tokenizer,
    GPTNeoXTokenizerFast,
    LlamaTokenizer,
    LlamaTokenizerFast,
    OPTForCausalLM,
    get_scheduler,
)

from dataset_processor import CHAT_TEMPLATES
from utils import (
    ArgumentParserPlus,
    clean_last_n_checkpoints,
    get_datasets,
    get_last_checkpoint_path,
    get_wandb_tags,
    is_beaker_job,
    maybe_get_beaker_config,
    maybe_use_ai2_hf_entity,
    maybe_use_ai2_wandb_entity,
    upload_metadata_to_hf,
)

logger = get_logger(__name__)

import pickle
from tqdm import tqdm

@dataclass
class FlatArguments:
    """
    Full arguments class for all fine-tuning jobs.
    """

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """The name of this experiment"""
    run_name: Optional[str] = None
    """A unique name of this run"""
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    peft_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The peft model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    tokenizer_revision: Optional[str] = field(
        default=None,
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    chat_template_name: str = field(
        default="tulu",
        metadata={
            "help": (
                f"The name of the chat template to use. "
                f"You can choose one of our pre-defined templates: {', '.join(CHAT_TEMPLATES.keys())}."
                f"Or, you can provide a tokenizer name or path here and we will apply its chat template."
            )
        },
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether to use flash attention in the model training"},
    )
    use_slow_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the slow tokenizer or not (which is then fast tokenizer)."},
    )
    model_revision: Optional[str] = field(
        default=None,
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. "
                "This option should only be set to `True` for repositories you trust and in which you "
                "have read the code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, "
                "then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_mixer: Optional[dict] = field(
        default=None, metadata={"help": "A dictionary of datasets (local or HF) to sample from."}
    )
    dataset_mixer_list: Optional[list[str]] = field(
        default=None, metadata={"help": "A list of datasets (local or HF) to sample from."}
    )
    dataset_mix_dir: Optional[str] = field(
        default=None, metadata={"help": "The directory to save the mixed dataset to disk."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a json/jsonl file)."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. "
                "Sequences longer than this will be truncated,"
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    add_bos: bool = field(
        default=False,
        metadata={
            "help": "Forcibly add bos token to the beginning of the input sequence."
            " Use only when tokenizer does not add bos token by default."
        },
    )
    clip_grad_norm: float = field(
        default=-1,
        metadata={"help": "Clip gradient norm. Not compatible with deepspeed (use deepspeed config instead)."},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "The initial learning rate for AdamW optimizer."},
    )
    logging_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Log the training loss and learning rate every logging_steps steps."},
    )
    lora_rank: int = field(
        default=64,
        metadata={"help": "The rank of lora."},
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": "The alpha parameter of lora."},
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout rate of lora modules."},
    )
    lr_scheduler_type: str = field(
        default="linear",
        metadata={
            "help": "The scheduler type to use for learning rate adjustment.",
            "choices": ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        },
    )
    num_train_epochs: int = field(
        default=2,
        metadata={"help": "Total number of training epochs to perform."},
    )
    output_dir: str = field(
        default="output/",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    per_device_train_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training."},
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "If True, will use LORA (low-rank parameter-efficient training) to train the model."},
    )
    use_qlora: bool = field(
        default=False,
        metadata={"help": "Use qLoRA training - initializes model in quantized form. Not compatible with deepspeed."},
    )
    use_8bit_optimizer: bool = field(
        default=False,
        metadata={"help": "Use 8bit optimizer from bitsandbytes. Not compatible with deepspeed."},
    )
    warmup_ratio: float = field(
        default=0.03,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."},
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay for AdamW if we apply some."},
    )
    timeout: int = field(
        default=1800,
        metadata={
            "help": "Timeout for the training process in seconds."
            "Useful if tokenization process is long. Default is 1800 seconds (30 minutes)."
        },
    )
    reduce_loss: str = field(
        default="sum",
        metadata={
            "help": "How to reduce loss over tokens. Options are 'mean' or 'sum'."
            "Using 'sum' can improve chat model performance."
        },
    )
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": "Entity to use for logging to wandb."},
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "If the training should continue from a checkpoint folder."},
    )
    with_tracking: bool = field(
        default=False,
        metadata={"help": "Whether to enable experiment trackers for logging."},
    )
    report_to: Union[str, List[str]] = field(
        default="all",
        metadata={
            "help": "The integration(s) to report results and logs to. "
            "Can be a single string or a list of strings. "
            "Options are 'tensorboard', 'wandb', 'comet_ml', 'clearml', or 'all'. "
            "Specify multiple by listing them: e.g., ['tensorboard', 'wandb']"
        },
    )
    save_to_hub: Optional[str] = field(
        default=None,
        metadata={"help": "Save the model to the Hub under this name. E.g allenai/your-model"},
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Turn on gradient checkpointing. Saves memory but slows training."},
    )
    max_train_steps: Optional[int] = field(
        default=None,
        metadata={"help": "If set, overrides the number of training steps. Otherwise, num_train_epochs is used."},
    )
    seed: int = field(default=42, metadata={"help": "Random seed for initialization and dataset shuffling."})
    checkpointing_steps: Optional[str] = field(
        default=None,
        metadata={
            "help": "Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch."  # noqa
        },
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": "Overwrite the content of the output directory. Means that resumption will always start from scratch."
        },
    )
    keep_last_n_checkpoints: int = field(
        default=5,
        metadata={"help": "How many checkpoints to keep in the output directory. -1 for all."},
    )
    fused_optimizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use fused AdamW or not.",
        },
    )
    load_balancing_loss: bool = field(
        default=False,
        metadata={
            "help": "Whether to include a load balancing loss (for OLMoE) or not.",
        },
    )
    load_balancing_weight: float = field(
        default=0.5,
        metadata={"help": "Weight for load balancing loss if applicable."},
    )
    push_to_hub: bool = None
    """Whether to upload the saved model to huggingface"""
    hf_entity: Optional[str] = None
    """The user or org name of the model repository from the Hugging Face Hub"""
    hf_repo_id: Optional[str] = None
    """The id of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_revision: Optional[str] = None
    """The revision of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_url: Optional[str] = None
    """The url of the saved model in the Hugging Face Hub (will be autoset)"""
    try_launch_beaker_eval_jobs: bool = False
    """Whether to launch beaker evaluation jobs after training"""
    hf_metadata_dataset: Optional[str] = None
    """What dataset to upload the metadata to. If unset, don't upload metadata"""

    def __post_init__(self):
        if self.reduce_loss not in ["mean", "sum"]:
            raise ValueError("reduce_loss must be either 'mean' or 'sum'")
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.dataset_mixer is None
            and self.dataset_mixer_list is None
        ):
            raise ValueError("Need either a dataset name, dataset mixer, or a training file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["json", "jsonl"], "`train_file` should be a json or a jsonl file."
        if (
            (self.dataset_name is not None and (self.dataset_mixer is not None or self.dataset_mixer_list is not None))
            or (self.dataset_name is not None and self.train_file is not None)
            or (
                (self.dataset_mixer is not None or self.dataset_mixer_list is not None) and self.train_file is not None
            )
            or (self.dataset_mixer is not None and self.dataset_mixer_list is not None)
        ):
            raise ValueError("Cannot provide two dataset selection mechanisms.")
        if self.try_launch_beaker_eval_jobs and not self.push_to_hub:
            raise ValueError("Cannot launch Beaker evaluation jobs without pushing to the Hub.")


def encode_sft_example(example, tokenizer, max_seq_length):
    """
    This function encodes a single example into a format that can be used for sft training.
    Here, we assume each example has a 'messages' field. Each message in it is a dict with 'role' and 'content' fields.
    We use the `apply_chat_template` function from the tokenizer to tokenize the messages and prepare the input and label tensors.
    """
    index = example["index"]
    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")
    input_ids = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=True,
        return_tensors="pt",
        padding = False,
        truncation=True,
        max_length=max_seq_length,
        add_generation_prompt=False,
    )
    labels = input_ids.clone()
    
    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            # we calculate the start index of this non-assistant message
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer.apply_chat_template(
                    conversation=messages[:message_idx],  # here marks the end of the previous messages
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=False,
                ).shape[1]
            # next, we calculate the end index of this non-assistant message
            if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
                # for intermediate messages that follow with an assistant message, we need to
                # set `add_generation_prompt=True` to avoid the assistant generation prefix being included in the loss
                # (e.g., `<|assistant|>`)
                message_end_idx = tokenizer.apply_chat_template(
                    conversation=messages[: message_idx + 1],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=True,
                ).shape[1]
            else:
                # for the last message or the message that doesn't follow with an assistant message,
                # we don't need to add the assistant generation prefix
                message_end_idx = tokenizer.apply_chat_template(
                    conversation=messages[: message_idx + 1],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=False,
                ).shape[1]
            # set the label to -100 for the non-assistant part
            labels[:, message_start_idx:message_end_idx] = -100
            if max_seq_length and message_end_idx >= max_seq_length:
                break
    attention_mask = torch.ones_like(input_ids)
    return {
        "index": torch.tensor(index),
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }

class CustomDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    def __init__(self, tokenizer, model, padding="longest"):
        super().__init__(tokenizer, model, padding=padding)

    def __call__(self, features):
        # 分离出id字段和需要填充的字段
        id_field = [feature.pop('index', None) for feature in features]
        
        # 调用父类的__call__方法来处理填充和截断
        batch = super().__call__(features)
        
        # 将id字段重新添加到批次中
        # 假设id_field是一个列表，包含了每个样本的id
        batch['index'] = id_field
        
        return batch

def main(args: FlatArguments):
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    args.run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"

    # if you get timeouts (e.g. due to long tokenization) increase this.
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=args.timeout))
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
        )
    elif args.dataset_mixer is not None:
        # mixing datasets via config
        raw_datasets = get_datasets(
            args.dataset_mixer,
            configs=args.dataset_config_name,
            splits=["train"],
            save_data_dir=args.dataset_mix_dir if accelerator.is_main_process else None,
            columns_to_keep=["messages"],
        )
    elif args.dataset_mixer_list is not None:
        # mixing datasets via config
        raw_datasets = get_datasets(
            args.dataset_mixer_list,
            configs=args.dataset_config_name,
            splits=["train"],
            save_data_dir=args.dataset_mix_dir if accelerator.is_main_process else None,
            columns_to_keep=["messages"],
        )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            **dataset_args,
        )

    # Load pretrained model and tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            revision=args.model_revision,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            revision=args.model_revision,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )

    tokenizer_revision = args.model_revision if args.tokenizer_revision is None else args.tokenizer_revision
    if tokenizer_revision != args.model_revision:
        # Warn user if tokenizer and model use different revisions; this is an unusual
        # use case.
        warning = f"""Requested tokenizer revision `{tokenizer_revision}` is different
                   from the model revision `{args.model_revision}`."""
        logger.warning(warning)

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=tokenizer_revision,
            trust_remote_code=args.trust_remote_code,
            use_fast=not args.use_slow_tokenizer,
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            revision=tokenizer_revision,
            trust_remote_code=args.trust_remote_code,
            use_fast=not args.use_slow_tokenizer,
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            revision=args.model_revision,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            trust_remote_code=args.trust_remote_code,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
        )

    # no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map
    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
            }
        )
        assert num_added_tokens in [
            0,
            1,
        ], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
    elif isinstance(tokenizer, GPTNeoXTokenizerFast):
        # OLMo newer models use this tokenizer
        if tokenizer.bos_token is None:
            tokenizer.bos_token = tokenizer.eos_token
            assert (
                args.add_bos
            ), "For OLMo with GPTNeoX, you must add bos token to the beginning of the input sequence."
        # else, pythia / other models
        else:
            num_added_tokens = tokenizer.add_special_tokens(
                {
                    "pad_token": "<pad>",
                }
            )
            assert num_added_tokens == 1, "GPTNeoXTokenizer should only add one special token - the pad_token."
    elif isinstance(tokenizer, GPT2Tokenizer) and isinstance(model, OPTForCausalLM):
        num_added_tokens = tokenizer.add_special_tokens({"unk_token": "<unk>"})
    elif isinstance(tokenizer, transformers.PreTrainedTokenizerFast) and tokenizer.pad_token is None:
        num_added_tokens = tokenizer.add_special_tokens({"pad_token": "<pad>"})
        assert num_added_tokens == 1, "We detected no padding token but add_special_tokens did not add one."

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    # gather deepspeed to get "real" embedding size
    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]
    # resize does its own gather
    if len(tokenizer) > embedding_size:
        # pad to multiple for tensor cores.
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    # update embedding size after resizing for sum loss
    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]

    # set the tokenizer chat template to the training format
    # this will be used for encoding the training examples
    # and saved together with the tokenizer to be used later.
    if args.chat_template_name in CHAT_TEMPLATES:
        tokenizer.chat_template = CHAT_TEMPLATES[args.chat_template_name]
    else:
        try:
            tokenizer.chat_template = AutoTokenizer.from_pretrained(args.chat_template_name).chat_template
        except Exception:
            raise ValueError(f"Could not find chat template for {args.chat_template_name}.")

    if args.add_bos:
        if tokenizer.chat_template.startswith("{{ bos_token }}") or (
            tokenizer.bos_token is not None and tokenizer.chat_template.startswith(tokenizer.bos_token)
        ):
            raise ValueError(
                "You specified add_bos=True, but the chat template already has a bos_token at the beginning."
            )
        # also add bos in the chat template if not already there
        tokenizer.chat_template = "{{ bos_token }}" + tokenizer.chat_template

    #model = PeftModel.from_pretrained(model, args.peft_model_name_or_path, device_map='cuda')
    model.cuda()
    model.eval()

    accelerator = Accelerator()
    
    train_dataset = raw_datasets["train"]
    
    train_dataset = train_dataset.add_column("index", range(len(train_dataset)))
    
    # debugging tool for fewer samples
    if args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), args.max_train_samples)
        logger.info(f"Limiting training samples to {max_train_samples} from {len(train_dataset)}.")
        train_dataset = train_dataset.select(range(max_train_samples))

    
    train_dataset = train_dataset.map(
        partial(encode_sft_example, tokenizer=tokenizer, max_seq_length=args.max_seq_length),
        batched=False,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        remove_columns=[
            name for name in train_dataset.column_names if name not in ["input_ids", "labels", "attention_mask", 'index']
        ],
        desc="Tokenizing and reformatting instruction data",
    )
    train_dataset.set_format(type="pt")
    train_dataset = train_dataset.filter(lambda example: (example["labels"] != -100).any())

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=False,
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="max_length", max_length = args.max_seq_length),
        batch_size=args.per_device_train_batch_size,
        num_workers=4,
        pin_memory=True,
    )
    # Train!
    all_samples = []

    
    
    for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        index = batch.pop("index")
        batch = {k: v.to('cuda') for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch, use_cache=False)
        logits = outputs.logits
        labels = batch["labels"]
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
        shift_logits = shift_logits.view(-1, embedding_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels) / (labels >= 0).sum().item()
        # Tracking loss
        
        token_level_loss = torch.nn.CrossEntropyLoss(reduction="none")(shift_logits, shift_labels)
        token_level_loss = token_level_loss.reshape(logits.shape[0], -1)
        for i in range(logits.shape[0]):
            sample_data = {
                'input_ids': batch['input_ids'][i].detach().cpu().numpy(),
                'attention_mask': batch['attention_mask'][i].detach().cpu().numpy(),
                'labels': batch['labels'][i].detach().cpu().numpy(),
                'loss': token_level_loss[i].detach().cpu().numpy(),
                'index': index[i].detach().cpu().numpy(),
                'logits': logits[i].detach().cpu().numpy(),
            }
                        
            all_samples.append(sample_data)
        
        for i, sample in enumerate(all_samples):
            file_path = os.path.join(args.output_dir, f'test_info_{index[i].detach().cpu().numpy()}.pkl')
            with open(file_path, 'wb') as f:
                pickle.dump(sample, f)
        all_samples = []


if __name__ == "__main__":
    parser = ArgumentParserPlus((FlatArguments))
    args = parser.parse()
    main(args)
