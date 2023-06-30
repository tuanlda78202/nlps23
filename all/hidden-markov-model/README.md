## EECS 738 Project 2: Hidden Markov Model by Matthew Taylor

### Overview
The objective of this project was to create a hidden Markov model (HMM) capable of generating and predicting text. This task was accomplished by implementing the [Baum-Welch](https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm) and [Viterbi](https://en.wikipedia.org/wiki/Viterbi_algorithm) algorithms. This HMM was trained using a [data set](https://www.kaggle.com/snap/amazon-fine-food-reviews) containing user reviews of food products on Amazon.

### Approach
I was attracted to this data set for many reasons. Its large number of observations provided ample training data and most of the sentences seemed to have a similar structure. I assumed the model would easily be able to learn and reproduce this repeated sentence structure after observing it so many times. After some light cleaning (removing strange characters, HTML tags, and punctuation) and restructuring (only a single sentence on every line), the data was ready to be consumed.

As for the general structure of the HMM, I had always planned on using 8 hidden states. My intention was to have each state represent one of the major parts of speech: nouns, pronouns, verbs, adjectives, adverbs, prepositions, conjunctions, and articles. After the model's structure was decided, I began training the model.

Hidden Markov models can be entirely described with three parameters. Those parameters are the model's initial probabilities, transition probabilities, and emission probabilities. Training a HMM is the process of obtaining these parameters and a standard algorithm for doing so is the Baum-Welch algorithm. Note: due to an incredibly long training time, only the first 2,500 reviews of the data set were used.

Once the model was trained, it could begin generating new text using simple random number generation and its initial, transition, and emission probabilities. The model could also predict the next words in a given sequence using the Viterbi algorithm in conjunction with the previously described text generation process.

### How To Run
This project was written in Python 3.7.2 and relies on two modules, [numpy](http://www.numpy.org/) and [nltk](https://www.nltk.org/). To install these modules, simply run this command:
```
pip3 install numpy nltk
```
Then, an nltk resource named punkt must be installed. This can be done by doing the following:
```
python3
>>> import nltk
>>> nltk.download('punkt')
```
Once the dependencies are installed, text can be generated using this command:
```
python3 generate.py
```
And text prediction can be performed with:
```
python3 predict.py
```
Both of these Python scripts ask the user for the required information during execution, so no additional command line parameters are necessary.

### How To Train
**Note: The model in this repo is already trained. Additional training is not required. Training this model takes about 1 hour.**
If you choose to retrain this model, run the following command:
```
python3 train.py
```
If you would like to change the number of hidden states the HMM uses, you must alter the contents of `train.py`. If you do this, you must also change the number of states in `generate.py` and `predict.py` in order for those scripts to function properly.

### Results
Sample output of the `generate.py` and `predict.py` scripts is shown below.
```
>python3 generate.py
Number of words to generate: 8
outside and i use it for ingredient coffee

>python3 predict.py
Enter a sequence of words: i like
Number of words to predict: 5
it chocolate or a little
```

As you can probably guess, most of the text produced by this model is nonsense. However, I implore you to look beyond the meaning of the words themselves. I ask that you try generating text for yourself and that you inspect the general sentence structure of the output (i.e. look for patterns like: article, noun, adverb, verb), as this was what I designed this model to do.

### Possible Optimizations and Future Work
As previously stated, this model takes a very long time to train. Even with a relatively small number of inputs, convergence was not achieved and the Baum-Welch algorithm was stopped after 100 iterations. Given more time to improve this project, I would try pruning the emission probabilities matrix, which is by far the largest of the three parameters. I would try to remove observations that occurred so infrequently, that the probability of seeing the finished model emit said observation would be essentially 0. This would reduce both the time and space complexity of the training process.

Furthermore, I might reconsider the data structure used to house the emission probabilities matrix. In this project, I implemented it using a dictionary. Although I'm not particularly familiar with the performance of this data structure in Python, I'm curious to see how another, more versatile data structure would have performed.

Finally, I would like to have reworked the general structure of the HMM. Using a state for each part of speech seemed like a good idea at the time, but produced lackluster results in the end. Given another chance, I would be tempted to treat each individual word as its own state and focus more on each word's preceding and following words. I believe this strategy might create a model that generates more natural sentences.
