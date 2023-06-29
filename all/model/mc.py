import random
import string


# The length of lines generated based on HMM is restricted to this value.
# It is a safeguard to protect the code from a potential infinite loop
MAX_LINE_LENGTH = 8

# This is used to check the equality of two floating point values. If
# their difference is less than this value, they are considered equal
FLOAT_DELTA = 0.0001

# This special string is used to mark the end of a sentence
END_TOKEN = 'endl'

# Remove white-spaces and punctuations from a line and convert it
# into a list of tokens
def tokenize (line):
    baseline = line.strip ().lower ()
    tokens = ''.join ([x for x in baseline if x not in string.punctuation]).split ()
    return tokens

# Add a pairing to dictionary
def insert_link (dictionary, key, value, debug = False):
    if key not in dictionary:
        dictionary [key] = []
    if debug: print(key, dictionary [key])
    dictionary [key].append (value)

# Convert list to probability values
def to_probability (chain):
    frequencies = {}
    probabilities = {}
    num_of_words = len (chain)

    for word in chain:
        frequencies [word] = frequencies.get (word, 0) + 1

    for word, frequency in frequencies.items ():
        probabilities [word] = round (float (frequency) / num_of_words, 3)

    return probabilities

def build_markov_model (corpus, first_order_markov_chain, second_order_markov_chain):
    # This is a dictionary of words which have been used to
    # start a line in Shakespeare's plays
    words = []

    for line in corpus:
        tokens = tokenize (line)
        num_of_tokens = len (tokens)

        for idx in range (num_of_tokens):
            token = tokens [idx]

            if idx == 0:
                words.append (token)

                # We are not interested in the first word of a
                # line since nothing precedes it
                continue

            # Populate first-order markov chain
            last_token = tokens [idx - 1]
            insert_link (first_order_markov_chain, last_token, token)

            # The second word in a line can only have a first-level
            # markov chain since there is only a single word before it
            if idx == 1:
                continue

            # The last pair of word of a line is special. We want
            # to chain it with 'END'; to help in finishing a line
            # during predicitions
            if idx == num_of_tokens - 1:
                insert_link (second_order_markov_chain, (last_token, token), END_TOKEN)

            # Populate second-order markov chain
            second_last_token = tokens [idx - 2]
            insert_link (second_order_markov_chain, (second_last_token, last_token), token)

    # Convert first-order markov chain to probability values
    for word, chain in first_order_markov_chain.items ():
        first_order_markov_chain [word] = to_probability (chain)

    # Convert first-order markov chain to probability values
    for pair, chain in second_order_markov_chain.items ():
        second_order_markov_chain [pair] = to_probability (chain)

    print ('[STATUS] Successfully built Markov Model on Corpus!\n')
    return list (set (words))

# Pick next word from the second-order markov chain. It should be the
# highest probability one. If multiple such words exist, randomly pick one
def predict_next_word (key, dictionary, debug = False):
    max_probability = 0.0
    most_probable_words = []

    for next_word, probability in dictionary.items ():
        if probability > max_probability:
            max_probability = probability
            most_probable_words = [next_word]
        elif max_probability - probability < FLOAT_DELTA:
            most_probable_words.append (next_word)

    if debug: print (key, most_probable_words)
    return random.choice (most_probable_words)

# Randomly pick a word that can follow; from the first-order markov chain
def pick_next_word (key, dictionary, debug = False):
    if debug: print(dictionary)
    return random.choice (list(dictionary.keys()))

# Generate text based on corpus
def write_line (start_word, markov_chain_one, markov_chain_two):
    line = []
    word = start_word.lower ()

    if word not in markov_chain_one.keys ():
        return 0

    line.append (word)
    next_word = pick_next_word (start_word, markov_chain_one [start_word])
    line.append (next_word)

    n = 0
    while n < MAX_LINE_LENGTH:
        next_next_word = predict_next_word ((word, next_word), markov_chain_two [(word, next_word)])

        if next_next_word == END_TOKEN:
            return ' '.join (line)

        word = next_word
        next_word = next_next_word
        line.append (next_next_word)
        n += 1

# Write a Shakespeare play of given length
def write_play (hints, mc1, mc2):
    string_generated = ""
    for word in hints:
        line = write_line (word, mc1, mc2)
        if (line):
          line += "\n"
          string_generated += line
    return string_generated

def predict_next (sequence, mc1, mc2):
    # Sanity checks
    sequence = sequence.strip ()
    if (sequence == ""):
        raise ValueError('Sequence must not be an empty string. Please retry!')

    tokens = tokenize (sequence)
    line = ''
    for token in reversed (tokens):
        line = write_line (token, mc1, mc2)

        if line:
            break

    return line + "\n"