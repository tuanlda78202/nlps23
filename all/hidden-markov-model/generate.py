import argparse
import pickle
from hmm import HMM

import spacy
import torch
from evaluate import load
from collections import defaultdict
from nltk.util import ngrams


def compute_perplexity(all_texts_list):
    torch.cuda.empty_cache()
    perplexity = load("perplexity", module_type="metric")
    # max sequence length and batch size are limited to 256 and 2, respectively, to avoid OOM bug
    resized_all_texts_list = [text[:256] for text in all_texts_list]
    results = perplexity.compute(
        predictions=resized_all_texts_list,
        model_id="vinai/bartpho-word",
        device="cuda",
        batch_size=2,
    )
    return results["mean_perplexity"]


def compute_wordcount(all_texts_list):
    wordcount = load("word_count")
    wordcount = wordcount.compute(data=all_texts_list)
    return wordcount["unique_words"]


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
        metrics[f"{n}gram_repitition"] = 1 - len(ngram_sets[n]) / ngram_counts[n]
    diversity = 1
    for val in metrics.values():
        diversity *= 1 - val
    metrics["diversity"] = diversity
    return metrics


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Vietnamese Poem Generator with HMM")

    # It takes 1 argument: number of tokens to generate
    # Example: python generate.py -n 256
    args.add_argument(
        "-n",
        "--number_of_tokens",
        type=int,
        default=256,
        help="Number of tokens to generate",
    )

    n = args.parse_args().number_of_tokens

    # load trained model
    hmm_parameters = pickle.load(open("pickle/8chu_hmm_parameters_final_1.p", "rb"))
    transition_prob = hmm_parameters[0]
    emission_prob = hmm_parameters[1]
    initial_prob = hmm_parameters[2]
    model = HMM(n_hidden_states=8, a=transition_prob, b=emission_prob, pi=initial_prob)
    model.generate(n)
    # print(emission_prob[0])
    # print(sum(emission_prob[0].values()))

    # print(transition_prob)
    # print(emission_prob)
    # print(initial_prob)
    """ list_of_string = []
    # generate words
    n = 256
    for i in range(300):
        generate = model.generate(n)
        list_of_string.append(generate)
    list_of_string = [i.replace("\n", ".") for i in list_of_string]
    perplexity = compute_perplexity(list_of_string)
    wordcount = compute_wordcount(list_of_string)
    diversity = compute_diversity(list_of_string)
    print("Perplexity:", perplexity)
    print("Wordcount:", wordcount)
    print("Diversity:", diversity) """
