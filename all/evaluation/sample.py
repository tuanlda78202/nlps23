import torch
import os
import json


class AMSample:
    def __init__(self,
                 trainer,
                 tokenizer,
                 save_sample_dir,
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

    def generate(self, test_dataset):
        if not os.path.exists("experiments"):
            os.mkdir("experiments")

        outputs = self.model.generate(torch.Tensor(test_dataset["input_ids"]).squeeze().to("cuda").long(),
                                      do_sample=self.do_sample,
                                      max_length=self.max_length,
                                      repetition_penalty=self.repetition_penalty,
                                      top_k=self.top_k,
                                      top_p=self.top_p
                                      )
        detok_outputs = [self.tokenizer.decode(x, skip_special_tokens=True) for x in outputs]
        with open(f"experiments/{self.save_sample_dir}", "w") as f:
            for i in range(len(detok_outputs)):
                sample = {"source": self.tokenizer.decode(test_dataset["input_ids"][i]),
                          "target": detok_outputs[i],
                          "real": self.tokenizer.deocde(test_dataset["labels"][i])
                          }
                json.dump(sample, f, indent=4)
