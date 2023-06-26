import torch
import os
import json
from all.evaluation.metrics import *

class ARSample:
    def __init__(self,
                 trainer,
                 tokenizer,
                 save_sample_dir,
                 per_device_eval_batch_size,
                 do_sample=True,
                 max_length=100,
                 repetition_penalty=20.0,
                 top_k=50,
                 top_p=0.92,
                 ):
        self.model = trainer.model
        self.tokenizer = tokenizer
        self.save_sample_dir = save_sample_dir
        self.do_sample = do_sample
        self.max_length = max_length
        self.repetition_penalty = repetition_penalty
        self.top_k = top_k
        self.top_p = top_p
        self.per_device_eval_batch_size = per_device_eval_batch_size

    def generate(self, test_dataset):
        if not os.path.exists("experiments"):
            os.mkdir("experiments")
        samples = {
            "source": [],
            "target": [],
            "real": [],
            "complete_predict": [],
            "complete_actual": [],
        }
        metrics = {
            "perplexity": None,
            "wordcount": None,
            "diversity": None,
            "memorization": None,
            "mauve": None,
        }

        for i in range(0, len(test_dataset["input_ids"]), self.per_device_eval_batch_size):
            outputs = self.model.generate(
                torch.Tensor(test_dataset["input_ids"][:i + self.per_device_eval_batch_size]).squeeze().to(
                    "cuda").long(),
                do_sample=self.do_sample,
                max_length=self.max_length,
                repetition_penalty=self.repetition_penalty,
                top_k=self.top_k,
                top_p=self.top_p
                )
            detok_outputs = [self.tokenizer.decode(x, skip_special_tokens=True) for x in outputs]

            for i in range(len(detok_outputs)):
                source = self.tokenizer.decode(test_dataset["input_ids"][:i + self.per_device_eval_batch_size][i],
                                               skip_special_tokens=True)
                real = self.tokenizer.decode(test_dataset["labels"][:i + self.per_device_eval_batch_size][i],
                                             skip_special_tokens=True)
                samples["source"].append(source),
                samples["target"].append(detok_outputs[i])
                samples["real"].append(real)
                samples["complete_predict"].append(source.split(self.tokenizer.eos_token)[-1]+detok_outputs[i])
                samples["complete_actual"].append(source.split(self.tokenizer.eos_token)[-1]+real)
                # json.dump(sample, f, indent=4, ensure_ascii=False).encode("utf8")

        with open(f"experiments/{self.save_sample_dir}", "w", encoding="utf-8") as f:
            json_string = json.dumps(samples, ensure_ascii=False, indent=4)
            f.write(json_string)

        metrics["perplexity_gen"] = compute_perplexity(samples["complete_predict"])
        metrics["perplexity_real"] = compute_perplexity(samples["complete_actual"])
        metrics["wordcount"] = compute_wordcount(samples["complete_predict"])
        metrics["diversity"] = compute_diversity(samples["complete_predict"])
        metrics["memorization"] = compute_memorization(samples["complete_predict"], samples["complete_actual"])
        metrics["mauve"] = compute_mauve(samples["complete_predict"], samples["complete_actual"])

        if not os.path.exists("experiments/metrics_eval"):
            os.mkdir("experiments/metrics_eval")
        with open(f"experiments/metrics_eval/{self.save_sample_dir}", "w") as f:
            json.dump(metrics, f, indent=4)