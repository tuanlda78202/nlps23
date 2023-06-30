import pickle
from hmm import HMM

# load trained model
hmm_parameters = pickle.load(open('pickle/hmm_parameters.p', 'rb'))
transition_prob = hmm_parameters[0]
emission_prob = hmm_parameters[1]
initial_prob = hmm_parameters[2]
model = HMM(n_hidden_states=8, a=transition_prob, b=emission_prob, pi=initial_prob)

# predict
sequence = input('Enter a sequence of words: ')
n = int(input('Number of words to predict: '))
model.predict(sequence, n)