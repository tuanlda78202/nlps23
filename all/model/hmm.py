import numpy as np
import pickle
import re
import copy
from numpy.random import choice
import sys


# resources:
# https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm
# https://en.wikipedia.org/wiki/Viterbi_algorithm

class HMM:

	# description: 	constructor
	#
	# param: 		n_hidden_states (int), number of hidden states
	# param: 		a (2d array of floats), transition probabilities
	# param: 		b (dict), emission probabilities
	# param: 		pi (array of floats), initial probabilities
	def __init__(self, n_hidden_states, a=None, b=None, pi=None):
		
		self.n_hidden_states = n_hidden_states
		self.a = a
		self.b = b
		self.pi = pi

		# assign initial probabilities randomly if none are given
		if a is None:
			self.a = np.ones((n_hidden_states, n_hidden_states))

			for i in range(n_hidden_states):
				self.a[i] = np.random.dirichlet(np.ones(n_hidden_states), size=1)

		if pi is None:
			self.pi = np.random.dirichlet(np.ones(n_hidden_states), size=1)[0]



	# description:	implementation of the Baum-Welch algorithm, automatically
	#              	finds transition, emission, and initial probabilities
	#
	# param: 		observations (list of strings), sentences from text corpus
	# param: 		max_iterations (int, default 100), number of iterations to
	# 				stop after if convergence has not been reached
	def train(self, observations, max_iterations=1):
		observations = [observation + "\n" for observation in observations]
		print(str(observations[0]))
		# get unique words
		print('Counting unique words... ', end='')
		sys.stdout.flush()
		#print(type(observations))
		uniques = self.get_uniques(observations)

		print('done')
		sys.stdout.flush()

		# instantiate emission probabilities with random values
		temp = np.ones((self.n_hidden_states, len(uniques)))

		for i in range(self.n_hidden_states):
			temp[i] = np.random.dirichlet(np.ones(len(uniques)), size=1)

		self.b = []
		for i in range(self.n_hidden_states):
			self.b.append(dict(zip(uniques, temp[i])))

		del temp

		# variables to hold previous probabilities, to check for convergence
		self.old_a = np.ones((len(self.a), len(self.a[0])))
		self.old_b = np.ones((len(self.b), len(self.b[0])))
		self.old_pi = np.ones(len(self.pi))

		current_iteration = 0

		print('Training model with max_iterations={}...'.format(max_iterations))

		while current_iteration < max_iterations and not self.converged():
			current_iteration += 1
			print('Iteration', current_iteration)

			self.xx = [] # array of xi values for each observation
			self.gg = [] # array of gamma values for each observation

			# get xi and gamma values for every sentence
			for observation in observations:
				#print(type(observations))
				sentence = re.split(r'(\W+)', observation)
				sentence = list(filter(lambda a: a != " ", sentence))
				sentence = list(filter(lambda a: a != "", sentence))
				#sentence = [x.strip(' ') for x in sentence]

				#sentence = sentence.remove("' '")
				#print(sentence)
				#print(sentence)
				#print(type(observations))
				self.forward(sentence)
				self.backward(sentence)
				self.xi(sentence)
				self.xx.append(self.x)
				self.gamma(sentence)
				self.gg.append(self.g)

			# record old probabilities
			self.save()

			# update probabilities
			self.update(observations)

		print('Training complete, ', end='')
		if current_iteration == max_iterations:
			print('maximum iterations reached')
		else:
			print('convergence reached')

		# preserve results
		hmm_parameters = (self.a, self.b, self.pi)
		pickle.dump(hmm_parameters, open('pickle/lucbat_hmm_parameters_test.p', 'wb'))
			


	# description:	generates a list of unique words given a text corpus and
	# 				a map indicating which words belong to which index
	#
	# param: 		observations (list of strings), text from corpus
	#
	# return: 		uniques (array of strings), list of unique words
	def get_uniques(self, observations):
		
		uniques = []

		# multiple observation version
		for sentence in observations:
			words = re.split(r'(\W+)', sentence)
			for word in words:
				if word not in uniques:
					uniques.append(word)
		#print(uniques)
		#uniques.append("\n")
		uniques = list(filter(lambda a: a != " ", uniques))
		uniques = list(filter(lambda a: a != "", uniques))
		return uniques

	

	# description:	checks whether or not the Baum-Welch algorithm has caused
	#				the probabilities (a, b, and pi) to converge
	#
	# return:		(boolean), True if matricies have converged, False if not
	def converged(self):
		if not np.allclose(self.a, self.old_a):
			return False

		temp_b = []
		temp_old_b = []
		for i in range(len(self.b)):
			temp_b.append(list(self.b[i].values()))
			temp_old_b.append(list(self.old_b[i].values()))

		if not np.allclose(temp_b, temp_old_b):
			return False

		if not np.allclose(self.pi, self.old_pi):
			return False

		return True



	# description:	the forward portion of the Baum-Welch algorithm
	#				finds alpha values
	#
	# param:		observations (list of strings), a single sentence
	def forward(self, observations):

		# instantiate alpha matrix
		#print(type(observations))
		#print(observations)
		self.alpha = np.zeros((self.n_hidden_states, len(observations)))

		# base case
		for i in range(self.n_hidden_states):
			self.alpha[i, 0] = self.pi[i] * self.b[i][observations[0]]

		# normalize to prevent underflow
		self.alpha[:, 0] = self.normalize(self.alpha[:, 0])

		# recursive step
		for t in range(len(observations) - 1):
			for i in range(self.n_hidden_states):
				for j in range(self.n_hidden_states):
					self.alpha[i, t + 1] += self.alpha[j, t] * self.a[j, i]
				self.alpha[i, t + 1] = self.b[i][observations[t + 1]] * self.alpha[i, t + 1]

			# normalize to prevent underflow
			self.alpha[:, t + 1] = self.normalize(self.alpha[:, t + 1])



	# description:	the backward portion of the Baum-Welch algorithm
	#				finds beta values
	#
	# param:		observations (list of strings), a single sentence
	def backward(self, observations):
		self.beta = np.zeros((self.n_hidden_states, len(observations)))
		self.beta[:, -1] = 1

		for t in range(len(observations) - 2, -1, -1):
			for i in range(self.n_hidden_states):
				for j in range(self.n_hidden_states):
					self.beta[i, t] += self.beta[j, t + 1] * self.a[i, j] * self.b[j][observations[t + 1]]



	# description:	calculates temporary Baum-Welch variable gamma
	#
	# param:		observations (list of strings), a single sentence
	#
	# formula:		https://wikimedia.org/api/rest_v1/media/math/render/svg/5ecf98a64ec3cdf949076c545223d982bff56748
	def gamma(self, observations):

		self.g = np.zeros((self.n_hidden_states, len(observations)))

		for t in range(len(observations)):
			d = 0
			for i in range(self.n_hidden_states):
				self.g[i, t] = self.alpha[i, t] * self.beta[i, t]
				d += self.g[i, t]

			# prevent division by zero
			if d == 0:
				self.g[:, t] = np.zeros(self.g[:, t].shape)
			else:
				self.g[:, t] = self.g[:, t] / d



	# description:	calculates temporary Baum-Welch variable xi
	#
	# param:		observations (string), text from corpus
	#
	# formula:		https://wikimedia.org/api/rest_v1/media/math/render/svg/8b34e96879a32c5a158de9dbbcc8916a55478f1a
	def xi(self, observations):
		
		self.x = np.zeros((self.n_hidden_states, self.n_hidden_states, len(observations)))

		for t in range(len(observations) - 1):
			d = 0
			for i in range(self.n_hidden_states):
				for j in range(self.n_hidden_states):
					self.x[i, j, t] = self.alpha[i, t] * self.a[i, j] * self.beta[j, t + 1] * self.b[j][observations[t + 1]]
					d += self.x[i, j, t]
			
			# prevent division by zero
			if d == 0:
				self.x[:, :, t] = np.zeros(self.x[:, :, t].shape)
			else:
				self.x[:, :, t] = self.x[:, :, t] / d



	# description:	saves current a, b, and pi, so we can check for convergence later
	def save(self):
		self.old_a = copy.deepcopy(self.a)
		self.old_b = copy.deepcopy(self.b)
		self.old_pi = copy.deepcopy(self.pi)



	# description:	updates a, b, and pi
	#
	# param:		observations (string), text from corpus
	def update(self, observations):

		# update pi
		# https://wikimedia.org/api/rest_v1/media/math/render/svg/4473f2856293081692fd1b786f8af92b249076aa
		for i in range(self.n_hidden_states):
			temp = 0
			for t in range(len(observations)):
				temp += self.gg[t][i][0]
			self.pi[i] = temp
		self.pi = self.normalize(self.pi)

		# update a
		# https://wikimedia.org/api/rest_v1/media/math/render/svg/5c65e4b98b0796692e8227702fec4e62d6525e02
		for i in range(self.n_hidden_states):
			d = 0
			for t in range(len(observations)):
				d += np.sum(self.gg[t][i][:-1])

			for j in range(self.n_hidden_states):
				n = 0
				for t in range(len(observations)):
					n += np.sum(self.xx[t][i][j][:-1])
				self.a[i][j] = n / d

		# update b
		# https://wikimedia.org/api/rest_v1/media/math/render/svg/eef4f2a348673bcb6cb7cd545b84e451898cb7bd
		# https://wikimedia.org/api/rest_v1/media/math/render/svg/95d9663d215f615e725e59fd1c8031d6b7a6da8f
		for i in range(self.n_hidden_states):
			temp = dict.fromkeys(self.b[0], 0)
			d = 0
			for t1 in range(len(observations)):
				current_sentence = re.split(r'(\W+)', observations[t1])
				current_sentence = list(filter(lambda a: a != " ", current_sentence))
				current_sentence = list(filter(lambda a: a != "",  current_sentence))
				d += np.sum(self.gg[t1][i])
				for t2 in range(len(current_sentence)):
					current_word = current_sentence[t2]
					temp.update({current_word: temp[current_word] + self.gg[t1][i][t2]})

			for key in temp.keys():
				temp.update({key: temp[key] / d})

			self.b[i].update(temp)



	# description:	normalizes a given probability distribution
	#
	# return:		(list), normalized distribution
	def normalize(self, dist):
		# avoid division by zero
		if np.count_nonzero(dist) == 0:
			return dist
		x = 1 / np.sum(dist)
		return [x * p for p in dist]

	

	# description:	generates the given number of words
	#
	# param:		n (int), number of words to generate
	def generate(self, n):
		if self.a is None or self.b is None or self.pi is None:
			print('You must train the model before you can generate text')
			return

		if n < 1:
			print('You must enter a number greater than 0')
			return

		# get most likely initial state
		state = choice(range(self.n_hidden_states), p=self.pi)

		s = ''

		for i in range(n):
			s += choice(list(self.b[state].keys()), p=list(self.b[state].values())) + ' '
			state = choice(range(self.n_hidden_states), p=self.a[state])

		print(s)



	# description:	predicts given number of words that
	# 				proceed a given sequence (Viterbi algorithm)
	#
	# param:		sequence (string), initial sequence
	# param:		n (int), number of words to generate
	def predict(self, sequence, n):
		if self.a is None or self.b is None or self.pi is None:
			print('You must train the model before you can generate text')
			return
		
		if n < 1:
			print('You must enter a number greater than 0')
			return
		
		s = re.split(r'(\W+)', sequence)
		s = list(filter(lambda a: a != " ", s))
		s = list(filter(lambda a: a != "", s))
		
		# begin Viterbi algorithm
		t1 = np.zeros((self.n_hidden_states, len(s)))
		t2 = np.zeros((self.n_hidden_states, len(s)))

		for i in range(self.n_hidden_states):
			t1[i, 0] = self.pi[i] * self.safe_emission(i, s[0])
			t2[i, 0] = 0

		for i in range(1, len(s)):
			for j in range(self.n_hidden_states):
				t = t1[:, i - 1] * self.a[:][j] * self.safe_emission(j, s[i])
				t1[j][i] = np.max(t)
				t2[j][i] = np.argmax(t)

		z = np.zeros(len(s))
		x = np.zeros(len(s))

		z[-1] = np.argmax(t1[:, -1])
		x[-1] = int(z[-1])

		for i in range(len(s) - 1, 0, -1):
			z[i - 1] = t2[int(z[i]), i]
			x[i - 1] = z[i - 1]

		# end Viterbi algorithm

		# at this point, x contains the path most likely
		# to have produced the given sequence

		# state that produced the last word in the observation
		# (state that we are 'currently' in)
		state = x[-1]

		output = ''

		for i in range(n):
			# get next state
			state = choice(range(self.n_hidden_states), p=self.a[int(state)])
			# get emission from next state
			output += choice(list(self.b[int(state)].keys()), p=list(self.b[int(state)].values())) + ' '

		# predicted next words
		print(output)



	# description:	function to safely access emission matrix
	#
	# param:		state (int), hidden state number
	# param:		word (string), value to get emission prob of
	#
	# return:		(float), emission probability of given word from given state
	def safe_emission(self, state, word):
		if word in self.b[state].keys():
			return self.b[state][word]
		
		return 0
