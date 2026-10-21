"""Fine-tune (LoRA) and run inference with Flan-T5 on the reformatted LAM:ECHR data,
then run inference on the test set. The training configuration follows the paper
(Flan-T5-XXL, context = 0). See the README table for the per-configuration parameters.
"""
from datasets import load_from_disk, DatasetDict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from argparse import ArgumentParser
import torch
import pandas as pd
import os


def preprocess_function(sample, padding="max_length", test=False):
    """Tokenize a batch for the encoder-decoder model.

    The T5 task prefix is prepended to every input paragraph and the in-text annotated
    paragraph is used as the target. Test inputs are not truncated. Padding tokens in the
    labels are replaced by -100 so they are ignored in the loss.
    """
    # add the T5 task prefix to the input
    inputs = ["annotate legal arguments: " + item for item in sample["original"]]

    # tokenize inputs (no truncation at test time)
    if test:
        model_inputs = tokenizer(inputs, padding=padding, truncation=False)
    else:
        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # tokenize the targets with the `text_target` keyword argument
    if test:
        labels = tokenizer(text_target=sample["annotated"], padding=padding, truncation=False)
    else:
        labels = tokenizer(text_target=sample["annotated"], max_length=max_target_length, padding=padding, truncation=True)

    # replace all pad token ids in the labels by -100 to ignore padding in the loss
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--dataset', default='data/windowed_dataset-replicate-LAM-0', type=str,
                        help='paragraph-level (context = 0) dataset to fine-tune on')
    parser.add_argument('--model', default='google/flan-t5-xxl', type=str)
    parser.add_argument('--logging', action='store_true')
    parser.add_argument('--output', default='outputs/models/flant5', type=str)
    parser.add_argument('--prediction_dir', default='outputs/predictions', type=str)
    parser.add_argument('--epochs', default=2, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_seq', default=512, type=int)
    parser.add_argument('--lora_rank', default=16, type=int)
    parser.add_argument('--lora_alpha', default=32, type=int)
    args = parser.parse_args()

    dataset_path = args.dataset
    model_name = args.model
    logging = args.logging
    output_dir = args.output
    prediction_dir = args.prediction_dir
    epochs = args.epochs
    batch_size = args.batch_size
    # Flan-T5 uses the same maximum length for the source and the target sequence
    max_source_length = args.max_seq
    max_target_length = args.max_seq
    rank = args.lora_rank
    alpha = args.lora_alpha

    if logging:
        os.environ["COMET_LOG_ASSETS"] = "True"
        from comet_ml import Experiment
        experiment = Experiment(
            api_key=os.environ["COMET_API_KEY"],
            project_name="",
            workspace="",
            disabled=not logging,
        )

    # load the paragraph-level, in-text annotated dataset
    ds = load_from_disk(dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenized_dataset = DatasetDict()
    tokenized_dataset['train'] = ds['train'].map(
        preprocess_function, batched=True,
        remove_columns=["file_index", "index", "file_name", "original", "annotated", "context", "positions"])
    tokenized_dataset['test'] = ds['test'].map(
        lambda x: preprocess_function(x, test=True), batched=True,
        remove_columns=["file_index", "index", "file_name", "original", "annotated", "context", "positions"])

    # LoRA adapter on the attention and feed-forward projections
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=['q', 'v', 'k', 'o', 'wo', 'wi_0', 'wi_1'],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )

    # 8-bit quantization to fit the XXL model on a single 80GB GPU
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, quantization_config=bnb_config, torch_dtype="auto", device_map="auto")
    model = prepare_model_for_kbit_training(model)

    # add the LoRA adapter
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model.gradient_checkpointing_enable({"use_reentrant": False})
    model.to("cuda")

    # data collator pads the inputs and the labels to a multiple of 8
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        optim="paged_adamw_8bit",  # used with (Q)LoRA
        per_device_train_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=2,
        learning_rate=1e-3,
        num_train_epochs=epochs,
        predict_with_generate=True,
        generation_max_length=max_target_length,
        fp16=False,
        include_tokens_per_second=True,
        report_to=["comet_ml"] if logging else 'none',
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=False)

    def save_final_predictions(preds):
        """Decode the generated and the gold sequences and save them as a CSV with the
        columns `target` (gold) and `text` (prediction), as expected by the evaluation."""
        logits, labels, metrics = preds
        print("test predictions metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v}")
        predicted_sentence = [tokenizer.decode([l for l in x if l != -100], skip_special_tokens=True) for x in logits]
        original_sentence = [tokenizer.decode([l for l in x if l != -100], skip_special_tokens=True) for x in labels]

        df = pd.DataFrame({'target': original_sentence, 'text': predicted_sentence})
        os.makedirs(prediction_dir, exist_ok=True)
        output_file = os.path.join(prediction_dir, f'predictions-{model_name.split("/")[-1]}.csv')
        df.to_csv(output_file)
        print(f"saved predictions to {output_file}")

    # inference on the test set (generate the annotated version of each input paragraph)
    final_preds = trainer.predict(tokenized_dataset['test'], max_length=max_target_length)
    save_final_predictions(final_preds)
