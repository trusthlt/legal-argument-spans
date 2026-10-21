"""Fine-tune (QLoRA) or run in-context learning with the Llama 3 models on the
reformatted LAM:ECHR data, then run inference on the test set. The experiment
configuration is selected with --version and --model (see the README table)."""
from argparse import ArgumentParser
import os
from datasets import load_from_disk
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from tqdm import tqdm
import pandas as pd

from utils import (LABEL_MAPPING, AG as ANNOTATION_GUIDELINES,
                   BEGIN_ANNOTATION_TOKEN, END_ANNOTATION_TOKEN, END_LABEL_TOKEN)

def add_instructions(example):
        instruction = "\nAnnotate the given ECtHR judgment with formal argument types. "
        instruction += f"Mark the arguments by inserting the tags {BEGIN_ANNOTATION_TOKEN} and {END_ANNOTATION_TOKEN}. After these tags, append the label for the argument type and close the label with {END_LABEL_TOKEN}."
        instruction += " Do not modify the original text otherwise and do not append additional information or explain anything."
        instruction += f" Only attach a label from the following list: {list(LABEL_MAPPING.values())}"
        example['instruction'] = instruction
        return example

def add_annotation_guidelines(example):
    # possibility to modify the AG
    agl = ANNOTATION_GUIDELINES
    example['AG'] = agl
    return example

def get_icl_samples(n=2):
    prefered_labels = ['Application to the concrete case']
    l0 = 0
    l2 = 0
    n_selected = 0
    selected_labels = []
    selected_samples = []
    icl_samples = []
    for train_sample in ds['train']:
        if n_selected >= n:
            break
        if (len(train_sample['labels']) == 0) & (l0 == 0):
            selected_samples.append(train_sample)
            l0 += 1
            n_selected += 1
            selected_labels += train_sample['labels']
        elif (len(train_sample['labels']) == 2) & (l2 == 0):
            selected_samples.append(train_sample)
            l2 += 1
            n_selected += 1
            selected_labels += train_sample['labels']
        elif (l2 > 0) & (l0 > 0):
            if all([l in selected_labels for l in prefered_labels]):
                # you may chose any label
                selected_samples.append(train_sample)
                n_selected += 1
                selected_labels += train_sample['labels']
            elif any([(l in prefered_labels) and (l not in selected_labels) for l in train_sample['labels']]):
                selected_samples.append(train_sample)
                n_selected += 1
                selected_labels += train_sample['labels']
                # stick to the prefered labels
    return selected_samples

def formatting_prompts_func(examples, test=False):
    """Build the chat-formatted prompt for each example depending on `version`
    (0 = paragraph only, 1 = ICL, 2/3 = added context, 4 = context + guidelines,
    5 = full document, 6 = full document + guidelines)."""
    instructions = examples["instruction"]
    inputs       = examples["original"]
    outputs      = examples["annotated"]
    guidelines   = examples["AG"]
    if not((version == 5) or (version == 6)):
        context  = examples["context"]
    else:
        context = ['' for _ in inputs]
    messages = []

    # create the correct conversations depending on the version (with and without guidelines and context)
    for instruction, inp, out, ag, con in zip(instructions, inputs, outputs, guidelines, context):
        system_prompt = "You are a helpful legal assistant that can annotate legal decisions. You follow the annotation guidelines and label carefully."
        msg = [{"role": "system", "content": f"{system_prompt}"}]
        if (version == 6):
            msg += [{"role": "user", "content": f"### Instruction:\n{instruction}\n\n### Guidelines:\n{ag}\n\n### Input:\n{inp}"}]
        elif (version == 1):
            if with_icl:
                for s in icl_samples:
                    msg += [{"role": "user", "content": f"### Instruction:\n{s['instruction']}\n\n### Input:\n{s['original']}"}]
                    msg += [{"role": "assistant", "content": f"{s['annotated']}"}]
            msg += [{"role": "user", "content": f"### Instruction:\n{instruction}\n\n### Input:\n{inp}"}]
        elif (version == 2) or (version == 3):
            msg += [{"role": "user", "content": f"### Instruction:\n{instruction}\n\n### Context:\n{con}\n\n### Input:\n{inp}"}]
        elif (version == 4):
            if with_icl:
                for s in icl_samples:
                    msg += [{"role": "user", "content": f"### Instruction:\n{s['instruction']}\n\n### Guidelines:\n{s['AG']}\n\n### Context:\n{s['context']}\n\n### Input:\n{s['original']}"}]
                    msg += [{"role": "assistant", "content": f"{s['annotated']}"}]
            msg += [{"role": "user", "content": f"### Instruction:\n{instruction}\n\n### Guidelines:\n{ag}\n\n### Context:\n{con}\n\n### Input:\n{inp}"}]
        else:
            msg += [{"role": "user", "content": f"### Instruction:\n{instruction}\n\n### Input:\n{inp}"}]

        if not test:
            msg += [{"role": "assistant", "content": f"{out}"}]
        messages += [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=test,)] 
    return {"text": messages,}

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--dataset', default=None, type=str,
                        help='path to a specific dataset to load; if omitted, the '
                             'dataset matching --version is used (see mapping below)')
    parser.add_argument('--model', required=True, type=str,
                        help='unsloth/Meta-Llama-3.1-8B-Instruct or unsloth/Llama-3.3-70B-Instruct-bnb-4bit')
    parser.add_argument('--logging', action='store_true')
    parser.add_argument('--output', default='outputs/models', type=str)
    parser.add_argument('--prediction_dir', default='outputs/predictions', type=str)
    parser.add_argument('--epochs', default=1, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--gac', default=8, type=int)
    parser.add_argument('--max_seq', default=4_000, type=int)
    parser.add_argument('--lora_rank', default=16, type=int)
    parser.add_argument('--lora_alpha', default=32, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--inference_only', action='store_true',
                        help='skip training and run inference only, loading --adapter '
                             '(or --model, if --adapter is not given) directly')
    parser.add_argument('--adapter', default=None, type=str,
                        help='path to a saved LoRA adapter to load for --inference_only')
    parser.add_argument('--version', default=0, type=int)
    args = parser.parse_args()

    dataset_path = args.dataset
    max_seq_length = args.max_seq
    epochs = args.epochs
    batch_size = args.batch_size
    gac = args.gac
    output_dir = args.output
    prediction_dir = args.prediction_dir
    logging = args.logging
    if logging:
        from comet_ml import Experiment
    model_name = args.model
    rank = args.lora_rank
    alpha = args.lora_alpha
    resume = args.resume
    version = args.version
    with_icl = False
    if (version == 1):
        with_icl = True
    
    # The different dataset reformulations have different maximum lengths (due to the context window)

    # version 0 = context size 0        # Max source length: 2654
    # version 1 = ICL with 4 samples
    # version 2 = context size 2        # Max source length: 3309
    # version 3 = context size 4        # Max source length: 3540
    # version 4 = context size 4 + AG   # Max source length: 5231
    # version 5 = full                  # Max source length: 266053
    # version 6 = full + AG             # Max source length: 267744

    # check which dataset is required
    if (version == 0) or (version == 1):
        dataset_path = 'data/windowed_dataset-replicate-LAM-0'
    elif (version == 2):
        dataset_path = 'data/windowed_dataset-replicate-LAM-2'
    elif (version == 3) or (version == 4):
        dataset_path = 'data/windowed_dataset-replicate-LAM-4'
    elif (version == 5) or (version == 6):
        dataset_path = 'data/full-dataset'

    # `--version` selects both the prompt/context scheme (see formatting_prompts_func)
    # and, by default, the matching context-window dataset above. An explicit
    # `--dataset` overrides the path (it must match the version's context scheme).
    if args.dataset is not None:
        dataset_path = args.dataset

    if logging:
        experiment = Experiment(
        api_key=os.environ["COMET_API_KEY"],
        project_name="",
        workspace="",
        disabled=not logging,
        )

    ds = load_from_disk(dataset_path)
    dtype = None
    # we only load the 70B model in 4bit to conserve VRAM and save training resources
    load_in_4bit = False
    if model_name == 'unsloth/Llama-3.3-70B-Instruct-bnb-4bit':
        load_in_4bit = True
    

    if args.inference_only:
        # load a previously saved adapter (or --model, if it's itself a saved checkpoint) directly
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = args.adapter if args.adapter is not None else model_name,
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
            fast_inference = True,
        )
    else:
        # load model and tokenizer
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name,
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
            fast_inference = True,
        )

        # load peft model
        model = FastLanguageModel.get_peft_model(
            model,
            r = rank,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
            lora_alpha = alpha,
            lora_dropout = 0,
            bias = "none",
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 42,
            use_rslora = True,
            loftq_config = None,
        )

    # add instructions to dataset
    ds['train'] = ds['train'].map(lambda x: add_instructions(x))
    ds['test'] = ds['test'].map(lambda x: add_instructions(x))

    # add annotation guidelines to dataset
    ds['train'] = ds['train'].map(lambda x: add_annotation_guidelines(x))
    ds['test'] = ds['test'].map(lambda x: add_annotation_guidelines(x))

    # shuffle 
    ds['train'] = ds['train'].shuffle(seed=42)

    icl_samples = get_icl_samples(4)

    EOS_TOKEN = tokenizer.eos_token

    # map the conversation format to the test set (needed for both training and inference-only)
    ds['test'] = ds['test'].map(lambda x: formatting_prompts_func(x, test=True), batched = True,)

    if not args.inference_only:
        ds['train'] = ds['train'].map(lambda x: formatting_prompts_func(x, test=False), batched = True, remove_columns=['labels'])

        # only for the full dataset, we don't train on documents longer than 20k to not go OOM, so we filter the large outliers
        if (version == 5) or (version == 6):
            train_length = 20_000
            ds['train'] = ds['train'].filter(lambda x: len(tokenizer(x['text'])[0]) < train_length)
            # there are enough samples remaining for training
            print(ds['train'])

        # setup the trainer
        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = ds['train'],
            dataset_text_field = "text",
            max_seq_length = max_seq_length,
            dataset_num_proc = 2,
            packing = False,
            args = TrainingArguments(
                per_device_train_batch_size = batch_size,
                gradient_accumulation_steps = gac,
                warmup_steps = 5,
                num_train_epochs = epochs,
                learning_rate = args.lr,
                fp16 = not is_bfloat16_supported(),
                bf16 = is_bfloat16_supported(),
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                seed = 42,
                output_dir = os.path.join(output_dir,f"{model_name.split('/')[-1]}-version-{version}"),
                report_to = ["comet_ml"] if logging else "none",
                remove_unused_columns=True,
                save_strategy='steps',
                save_steps = 20,
                save_total_limit=2,
            ),
        )

        # resume from checkpoint if there is a previous training run with the same configuration (ie if training crashed)
        train_stats = trainer.train(resume_from_checkpoint=resume)

    # inference on the test set
    FastLanguageModel.for_inference(model) # unsloth native version of inference optimization
    predictions = []
    # generate a new predictions file with model name and version
    os.makedirs(prediction_dir, exist_ok=True)
    output_file = os.path.join(prediction_dir, f'predictions-{model_name.split("/")[-1]}-v{version}.csv')

    inference_batch_size = 1
    for i, sample in enumerate(tqdm(ds['test'].iter(batch_size=inference_batch_size))):
        model_inputs = tokenizer(sample['text'], padding=True, return_tensors='pt').to('cuda')

        input_length = len(model_inputs[0])
        # give an estimate of maximum new tokens to reduce inference time
        max_new = int(len(tokenizer(sample['original'], padding=True, return_tensors='pt')[0]))
        max_new = max_new + max(25, int(max_new*0.2))

        generated_ids = model.generate(**model_inputs, max_new_tokens=max_new, use_cache=True, )
        # only decode the newly generated text to save time (contains a copy of the input)
        decoded = tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)

        for dec, tar in zip(decoded, sample['annotated']):
            predictions.append({'target': tar, 'text': dec})

    already_predicted = pd.DataFrame(predictions)
    already_predicted.to_csv(output_file)

