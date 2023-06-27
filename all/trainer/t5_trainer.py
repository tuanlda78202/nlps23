from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

class T5Trainer:
    def __init__(self):

        training_args = Seq2SeqTrainingArguments(
            output_dir=config.args,
            overwrite_output_dir=True,
            num_train_epochs=20,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            save_steps=1000,
            save_total_limit=2,
            warmup_steps=1000,
            logging_steps=100,
            report_to="wandb",
        )

        trainer = Seq2SeqTrainer(
            model,
            args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            # compute_metrics=compute_metrics
        )