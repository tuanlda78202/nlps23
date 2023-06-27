import pickle
import nltk
from all.model.hmm import HMM
import sys
import string

print('Cleaning text... ', end='')
sys.stdout.flush()
# get raw text from file
#f = open('text/amazon_reviews.txt')
#text = ' '.join(f.read().splitlines())
with open('text/tiny_data_7chu.txt', 'r') as f:
    sentences = [line.strip() for line in f]
#print(text)
#f.close()

# split text into sentences
#sentences = nltk.sent_tokenize(text)
#print(sentences)

lines_to_remove = []
for i in range(len(sentences)):

	# remove all punctuation and set all words to lowercase
	sentences[i] = sentences[i].translate(str.maketrans('', '', string.punctuation)).lower()

	# record which lines are blank, so those can be removed
	if sentences[i] == '':
		lines_to_remove = [i] + lines_to_remove

# remove blank lines
for i in lines_to_remove:
	del sentences[i]
for sentence in sentences:
	sentence += "<\n>"
print('done')
sys.stdout.flush()

# train model
model = HMM(n_hidden_states=8)
model.train(sentences)