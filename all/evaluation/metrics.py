import spacy
import numpy as np
import torch
from evaluate import load
from collections import defaultdict
from nltk.util import ngrams


def compute_perplexity(all_texts_list):
    torch.cuda.empty_cache()
    perplexity = load("perplexity", module_type="metric")
    # max sequence length and batch size are limited to 256 and 2, respectively, to avoid OOM bug
    resized_all_texts_list = [text[:256] for text in all_texts_list]
    results = perplexity.compute(predictions=resized_all_texts_list, model_id="vinai/bartpho-word", device='cuda', batch_size=2)
    return results['mean_perplexity']


def compute_wordcount(all_texts_list):
    wordcount = load("word_count")
    wordcount = wordcount.compute(data=all_texts_list)
    return wordcount['unique_words']


def compute_diversity(all_texts_list):
    ngram_range = [2, 3, 4]

    tokenizer = spacy.load("en_core_web_sm").tokenizer
    token_list = []
    for sentence in all_texts_list:
        token_list.append([str(token) for token in tokenizer(sentence)])
    ngram_sets = {}
    ngram_counts = defaultdict(int)

    metrics = {}
    for n in ngram_range:
        ngram_sets[n] = set()
        for tokens in token_list:
            ngram_sets[n].update(ngrams(tokens, n))
            ngram_counts[n] += len(list(ngrams(tokens, n)))
        metrics[f'{n}gram_repitition'] = (1 - len(ngram_sets[n]) / ngram_counts[n])
    diversity = 1
    for val in metrics.values():
        diversity *= (1 - val)
    metrics['diversity'] = diversity
    return metrics


def compute_memorization(all_texts_list, human_references, n=4):
    tokenizer = spacy.load("en_core_web_sm").tokenizer
    unique_four_grams = set()
    for sentence in human_references:
        unique_four_grams.update(ngrams([str(token) for token in tokenizer(sentence)], n))

    total = 0
    duplicate = 0
    for sentence in all_texts_list:
        four_grams = list(ngrams([str(token) for token in tokenizer(sentence)], n))
        total += len(four_grams)
        for four_gram in four_grams:
            if four_gram in unique_four_grams:
                duplicate += 1

    return duplicate / total


def compute_mauve(all_texts_list, human_references):
    torch.cuda.empty_cache()
    mauve = load("mauve")
    assert len(all_texts_list) == len(human_references)

    results = mauve.compute(predictions=all_texts_list, references=human_references, featurize_model_name="vinai/phobert-base",
                            max_text_length=256, device_id=0)

    return results.mauve

#
# def evaluation_loop(
#         self,
#         dataloader: DataLoader,
#         description: str,
#         prediction_loss_only: Optional[bool] = None,
#         ignore_keys: Optional[List[str]] = None,
#         metric_key_prefix: str = "eval",
# ) -> EvalLoopOutput:
#     """
#     Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.
#
#     Works both with or without labels.
#     """
#     args = self.args
#
#     prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only
#
#     # if eval is called w/o train, handle model prep here
#     if self.is_deepspeed_enabled and self.model_wrapped is self.model:
#         _, _ = deepspeed_init(self, num_training_steps=0, inference=True)
#
#     model = self._wrap_model(self.model, training=False, dataloader=dataloader)
#
#     if len(self.accelerator._models) == 0 and model is self.model:
#         model = (
#             self.accelerator.prepare(model)
#             if self.is_deepspeed_enabled
#             else self.accelerator.prepare_model(model, evaluation_mode=True)
#         )
#
#         if self.is_fsdp_enabled:
#             self.model = model
#
#         # for the rest of this function `model` is the outside model, whether it was wrapped or not
#         if model is not self.model:
#             self.model_wrapped = model
#
#         # backward compatibility
#         if self.is_deepspeed_enabled:
#             self.deepspeed = self.model_wrapped
#
#     # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
#     # while ``train`` is running, cast it to the right dtype first and then put on device
#     if not self.is_in_train:
#         if args.fp16_full_eval:
#             model = model.to(dtype=torch.float16, device=args.device)
#         elif args.bf16_full_eval:
#             model = model.to(dtype=torch.bfloat16, device=args.device)
#
#     batch_size = self.args.eval_batch_size
#
#     logger.info(f"***** Running {description} *****")
#     if has_length(dataloader):
#         logger.info(f"  Num examples = {self.num_examples(dataloader)}")
#     else:
#         logger.info("  Num examples: Unknown")
#     logger.info(f"  Batch size = {batch_size}")
#
#     model.eval()
#
#     self.callback_handler.eval_dataloader = dataloader
#     # Do this before wrapping.
#     eval_dataset = getattr(dataloader, "dataset", None)
#
#
#     if args.past_index >= 0:
#         self._past = None
#
#     # Initialize containers
#     # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
#     losses_host = None
#     preds_host = None
#     labels_host = None
#     inputs_host = None
#
#     # losses/preds/labels on CPU (final containers)
#     all_losses = None
#     all_preds = None
#     all_labels = None
#     all_inputs = None
#     # Will be useful when we have an iterable dataset so don't know its length.
#
#     observed_num_examples = 0
#     # Main evaluation loop
#     for step, inputs in enumerate(dataloader):
#         # Update the observed num examples
#         observed_batch_size = find_batch_size(inputs)
#         if observed_batch_size is not None:
#             observed_num_examples += observed_batch_size
#             # For batch samplers, batch_size is not known by the dataloader in advance.
#             if batch_size is None:
#                 batch_size = observed_batch_size
#
#         # Prediction step
#         loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
#         inputs_decode = self._prepare_input(inputs["input_ids"]) if args.include_inputs_for_metrics else None
#
#         # Update containers on host
#         if loss is not None:
#             losses = self._nested_gather(loss.repeat(batch_size))
#             losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
#         if labels is not None:
#             labels = self._pad_across_processes(labels)
#         if inputs_decode is not None:
#             inputs_decode = self._pad_across_processes(inputs_decode)
#             inputs_decode = self._nested_gather(inputs_decode)
#             inputs_host = (
#                 inputs_decode
#                 if inputs_host is None
#                 else nested_concat(inputs_host, inputs_decode, padding_index=-100)
#             )
#         if logits is not None:
#             logits = self._pad_across_processes(logits)
#             if self.preprocess_logits_for_metrics is not None:
#                 logits = self.preprocess_logits_for_metrics(logits, labels)
#             logits = self._nested_gather(logits)
#             preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
#         if labels is not None:
#             labels = self._nested_gather(labels)
#             labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
#         self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)
#
#         # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
#         if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
#             if losses_host is not None:
#                 losses = nested_numpify(losses_host)
#                 all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
#             if preds_host is not None:
#                 logits = nested_numpify(preds_host)
#                 all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
#             if inputs_host is not None:
#                 inputs_decode = nested_numpify(inputs_host)
#                 all_inputs = (
#                     inputs_decode
#                     if all_inputs is None
#                     else nested_concat(all_inputs, inputs_decode, padding_index=-100)
#                 )
#             if labels_host is not None:
#                 labels = nested_numpify(labels_host)
#                 all_labels = (
#                     labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
#                 )
#
#             # Set back to None to begin a new accumulation
#             losses_host, preds_host, inputs_host, labels_host = None, None, None, None
#
#     if args.past_index and hasattr(self, "_past"):
#         # Clean the state at the end of the evaluation loop
#         delattr(self, "_past")
#
#     # Gather all remaining tensors and put them back on the CPU
#     if losses_host is not None:
#         losses = nested_numpify(losses_host)
#         all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
#     if preds_host is not None:
#         logits = nested_numpify(preds_host)
#         all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
#     if inputs_host is not None:
#         inputs_decode = nested_numpify(inputs_host)
#         all_inputs = (
#             inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
#         )
#     if labels_host is not None:
#         labels = nested_numpify(labels_host)
#         all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
#
#     # Number of samples
#     if has_length(eval_dataset):
#         num_samples = len(eval_dataset)
#     # The instance check is weird and does not actually check for the type, but whether the dataset has the right
#     # methods. Therefore we need to make sure it also has the attribute.
#     elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
#         num_samples = eval_dataset.num_examples
#     else:
#         if has_length(dataloader):
#             num_samples = self.num_examples(dataloader)
#         else:  # both len(dataloader.dataset) and len(dataloader) fail
#             num_samples = observed_num_examples
#     if num_samples == 0 and observed_num_examples > 0:
#         num_samples = observed_num_examples
#
#     # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
#     # samplers has been rounded to a multiple of batch_size, so we truncate.
#     if all_losses is not None:
#         all_losses = all_losses[:num_samples]
#     if all_preds is not None:
#         all_preds = nested_truncate(all_preds, num_samples)
#     if all_labels is not None:
#         all_labels = nested_truncate(all_labels, num_samples)
#     if all_inputs is not None:
#         all_inputs = nested_truncate(all_inputs, num_samples)
#
#     # Metrics!
#     if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
#         if args.include_inputs_for_metrics:
#             metrics = self.compute_metrics(
#                 EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
#             )
#         else:
#             metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
#     else:
#         metrics = {}
#
#     # To be JSON-serializable, we need to remove numpy types or zero-d tensors
#     metrics = denumpify_detensorize(metrics)
#
#     if all_losses is not None:
#         metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
#     if hasattr(self, "jit_compilation_time"):
#         metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time
#
#     # Prefix all keys with metric_key_prefix + '_'
#     for key in list(metrics.keys()):
#         if not key.startswith(f"{metric_key_prefix}_"):
#             metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
#
#     return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
#
