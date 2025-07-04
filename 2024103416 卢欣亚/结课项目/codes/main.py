

import logging
import os
import sys
import json

import numpy as np
from datasets import load_dataset
import jieba
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch
from torch.utils.data import ConcatDataset

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.utils import PaddingStrategy
from trainer_seq2seq import Seq2SeqTrainer

from arguments import ModelArguments, DataTrainingArguments
from dataclasses import asdict
from data_collator import Collator

logger = logging.getLogger(__name__)


def build_qa_prompt(query: str):
    return (
        f"Question: {query}\n"
        "Answer: \n"
        "Let's think step by step.\n"
    )


def build_chat_input(tokenizer, query, prompt: str, history=None, role="user"):
    if history is None:
        history = []
    input_ids = []
    for item in history:
        content = item["content"]
        if item["role"] == "system" and "tools" in item:
            content = content + "\n" + json.dumps(item["tools"], indent=4, ensure_ascii=False)
        input_ids.extend(tokenizer.build_single_message(item["role"], item.get("metadata", ""), content))
    if prompt:
        query = query + ' ' + prompt
    input_ids.extend(tokenizer.build_single_message(role, "", query))
    input_ids.extend([tokenizer.get_command("<|assistant|>")])
    return tokenizer.batch_encode_plus([input_ids], return_tensors="pt", is_split_into_words=True)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    args = {
        "model_args": asdict(model_args),
        "data_args": asdict(data_args),
        "training_args": asdict(training_args),
    }
    os.makedirs(training_args.output_dir, mode=0o775, exist_ok=True)
    with open(os.path.join(training_args.output_dir, "arguments.json"), "w", encoding="utf-8") as fout:
        json.dump(args, fout, ensure_ascii=False, indent=4)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load dataset
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.train_problem_solving_file is not None:
        data_files["problem_solving_train"] = data_args.train_problem_solving_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
    
    raw_datasets = load_dataset(
        path='json',
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )
    config.pre_seq_len = model_args.pre_seq_len
    config.prefix_projection = model_args.prefix_projection

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )

    if model_args.ptuning_checkpoint is not None:
        # Evaluation
        # Loading extra state dict of prefix encoder
        model = AutoModel.from_pretrained(
            model_args.model_name_or_path, config=config, trust_remote_code=True
        )
        if os.path.isfile(os.path.join(model_args.ptuning_checkpoint, "pytorch_model.bin")):
            prefix_state_dict = torch.load(
                os.path.join(model_args.ptuning_checkpoint, "pytorch_model.bin")
            )
        elif os.path.isfile(os.path.join(model_args.ptuning_checkpoint, "model.safetensors")):
            from safetensors import safe_open
            prefix_state_dict = {}
            with safe_open(os.path.join(model_args.ptuning_checkpoint, "model.safetensors"), framework="pt", device="cpu") as f:
                for key in f.keys():
                    prefix_state_dict[key] = f.get_tensor(key)
        else:
            raise FileNotFoundError("Checkpoint file does not exist.")

        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    else:
        model = AutoModel.from_pretrained(
            model_args.model_name_or_path, config=config, trust_remote_code=True
        )

    if model_args.quantization_bit is not None:
        print(f"Quantized to {model_args.quantization_bit} bit")
        model = model.quantize(model_args.quantization_bit)
    if model_args.pre_seq_len is not None:
        # P-tuning v2
        model = model.half()
        model.transformer.prefix_encoder.float()
    else:
        # Finetune
        model = model.half()

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info(
            "There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`."
        )
        return

    # Get the column names for input/target.
    prompt_column = data_args.prompt_column
    response_column = data_args.response_column
    history_column = data_args.history_column

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length

    def preprocess_function_eval(examples):
        inputs, targets = [], []
        for i in range(len(examples[prompt_column])):
            if examples[prompt_column][i] and examples[response_column][i]:
                query = examples[prompt_column][i]
                history = (
                    examples[history_column][i] if history_column is not None else None
                )
                if 'gsm8k' in data_args.test_file or 'mawps' in data_args.test_file:
                    pre_prompt = "Please analyze and solve the following problem step by step: "
                    # prompt = tokenizer.build_single_message("user", "", message=prefix + pre_prompt + query)
                    # prompt = [tokenizer.get_command("<|user|>")] + tokenizer.encode(
                    #     text=prefix + pre_prompt + query,
                    #     add_special_tokens=False,
                    #     truncation=True,
                    #     max_length=data_args.max_source_length,
                    # )
                    prompt = tokenizer.build_single_message('user', "", prefix + pre_prompt + query)
                    prompt += [tokenizer.get_command("<|assistant|>")]
                else:
                    prompt = tokenizer.build_prompt(query, history)
                inputs.append(prompt)
                targets.append(examples[response_column][i])

        if 'gsm8k' in data_args.test_file or 'mawps' in data_args.test_file:
            model_inputs = tokenizer.batch_encode_plus(
                inputs,
                is_split_into_words=True,
                padding=True,
                truncation=True,
                max_length=data_args.max_source_length
            )
            # pass
        else:
            inputs = [prefix + inp for inp in inputs]
            model_inputs = tokenizer(
                inputs,
                max_length=data_args.max_source_length,
                truncation=True,
                padding=True,
            )

        labels = tokenizer(
            text_target=targets, max_length=max_target_length, truncation=True
        )

        if data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label]
                for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def preprocess_function_train(examples):
        max_seq_length = data_args.max_source_length + data_args.max_target_length + 1

        model_inputs = {
            "input_ids": [],
            "labels": [],
        }
        for i in range(len(examples[prompt_column])):
            if examples[prompt_column][i] and examples[response_column][i]:
                query, answer = examples[prompt_column][i], examples[response_column][i]

                history = (
                    examples[history_column][i] if history_column is not None else None
                )
                prompt = tokenizer.build_prompt(query, history)

                prompt = prefix + prompt
                a_ids = tokenizer.encode(
                    text=prompt,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=data_args.max_source_length,
                )
                b_ids = tokenizer.encode(
                    text=answer,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=data_args.max_target_length,
                )

                context_length = len(a_ids)
                input_ids = a_ids + b_ids + [tokenizer.eos_token_id]
                labels = (
                    [tokenizer.pad_token_id] * context_length
                    + b_ids
                    + [tokenizer.eos_token_id]
                )

                pad_len = max_seq_length - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                labels = labels + [tokenizer.pad_token_id] * pad_len
                if data_args.ignore_pad_token_for_loss:
                    labels = [
                        (l if l != tokenizer.pad_token_id else -100) for l in labels
                    ]

                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)

        return model_inputs

    def preprocess_function_train_problem_solving(examples):
        max_seq_length = data_args.max_source_length + data_args.max_target_length + 1

        model_inputs = {
            "input_ids": [],
            "labels": [],
        }
        for i in range(len(examples[prompt_column])):
            if examples[prompt_column][i] and examples[response_column][i]:
                query, answer = examples[prompt_column][i], examples[response_column][i]

                history = (
                    examples[history_column][i] if history_column is not None else None
                )
                # pre_prompt = "Please analyze and solve the following problem step by step: "
                # prompt = " Let's think step by step."
                # prompt = tokenizer.build_single_message("user", "", message=prefix + query + prompt)
                # prompt = [tokenizer.get_command("<|user|>")] + tokenizer.encode(
                #     text=prefix + query,
                #     add_special_tokens=False,
                #     truncation=True,
                #     max_length=data_args.max_source_length,
                # )
                prompt = tokenizer.build_single_message("user", "", prefix + query)
                prompt += [tokenizer.get_command("<|assistant|>")]
                # prompt = tokenizer.build_prompt(query, history)
                a_ids = prompt
                b_ids = tokenizer.encode(
                    text=answer,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=data_args.max_target_length,
                )

                # a_ids = tokenizer.encode(
                #     text=prompt,
                #     add_special_tokens=True,
                #     truncation=True,
                #     max_length=data_args.max_source_length,
                # )
                # b_ids = tokenizer.encode(
                #     text=answer,
                #     add_special_tokens=False,
                #     truncation=True,
                #     max_length=data_args.max_target_length,
                # )

                context_length = len(a_ids)
                input_ids = a_ids + b_ids + [tokenizer.eos_token_id]
                labels = (
                    [tokenizer.pad_token_id] * context_length
                    + b_ids
                    + [tokenizer.eos_token_id]
                )

                pad_len = max_seq_length - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                labels = labels + [tokenizer.pad_token_id] * pad_len
                if data_args.ignore_pad_token_for_loss:
                    labels = [
                        (l if l != tokenizer.pad_token_id else -100) for l in labels
                    ]

                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)

        return model_inputs

    def print_dataset_example(example):
        print("input_ids", example["input_ids"])
        print("inputs", tokenizer.decode(example["input_ids"]))
        print("label_ids", example["labels"])
        print("labels", tokenizer.decode(example["labels"]))

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if "problem_solving_train" in raw_datasets:
            problem_solving_dataset = raw_datasets['problem_solving_train']
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function_train,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
            if "problem_solving_train" in raw_datasets:
                problem_solving_dataset = problem_solving_dataset.map(
                    preprocess_function_train_problem_solving,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on train dataset",
                )
                train_dataset = ConcatDataset([train_dataset, problem_solving_dataset])
        print_dataset_example(train_dataset[0])
        print_dataset_example(train_dataset[-1])

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(
            desc="validation dataset map pre-processing"
        ):
            eval_dataset = eval_dataset.map(
                preprocess_function_eval,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        print_dataset_example(eval_dataset[0])

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(
                len(predict_dataset), data_args.max_predict_samples
            )
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(
            desc="prediction dataset map pre-processing"
        ):
            predict_dataset = predict_dataset.map(
                preprocess_function_eval,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
        print_dataset_example(predict_dataset[0])

    # Data collator
    label_pad_token_id = (
        -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )
    collator_class = Collator if training_args.do_train else DataCollatorForSeq2Seq
    data_collator = collator_class(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
        padding=False,
    )

    # Metric
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))
            rouge = Rouge()
            try:
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]
                for k, v in result.items():
                    score_dict[k].append(round(v["f"] * 100, 4))
            except Exception:
                pass

            try:
                bleu_score = sentence_bleu(
                    [list(label)],
                    list(pred),
                    smoothing_function=SmoothingFunction().method3,
                )
                score_dict["bleu-4"].append(round(bleu_score * 100, 4))
            except Exception:
                score_dict["bleu-4"].append(0.0)

        for k, v in score_dict.items():
            score_dict[k] = float(np.mean(v))
        return score_dict

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    training_args.generation_num_beams = (
        data_args.num_beams
        if data_args.num_beams is not None
        else training_args.generation_num_beams
    )
    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=(
            compute_metrics if training_args.predict_with_generate else None
        ),
        save_changed=model_args.pre_seq_len is not None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        # elif last_checkpoint is not None:
        #     checkpoint = last_checkpoint
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_seq_length = data_args.max_source_length + data_args.max_target_length + 1
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            metric_key_prefix="eval",
            do_sample=False,
            top_p=0.7,
            max_length=max_seq_length,
            temperature=0.95,
        )
        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        predict_results = trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
            max_length=max_seq_length,
            do_sample=False,
            top_p=0.7,
            temperature=0.95,
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples
            if data_args.max_predict_samples is not None
            else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                predictions = [pred.strip() for pred in predictions]
                labels = tokenizer.batch_decode(
                    predict_results.label_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                labels = [label.strip() for label in labels]
                output_prediction_file = os.path.join(
                    training_args.output_dir, "generated_predictions.json"
                )
                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    for p, l in zip(predictions, labels):
                        res = json.dumps(
                            {"labels": l, "predict": p}, ensure_ascii=False
                        )
                        writer.write(f"{res}\n")
    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
