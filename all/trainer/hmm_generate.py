import pickle
from all.model.hmm import HMM

# load trained model
hmm_parameters = pickle.load(open('pickle/lucbat_hmm_parameters_test.p', 'rb'))
transition_prob = hmm_parameters[0]
emission_prob = hmm_parameters[1]
initial_prob = hmm_parameters[2]
model = HMM(n_hidden_states=8, a=transition_prob, b=emission_prob, pi=initial_prob)
print(emission_prob[0])
print(sum(emission_prob[0].values()))

print(transition_prob)
#print(emission_prob)
print(initial_prob)

# generate words
n = int(input('Number of words to generate: '))
model.generate(n)