########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of set 5. Once each part is implemented, you can simply
# execute the related problem scripts (e.g. run 'python 2G.py') to quickly
# see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding. If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!

import random

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states. 即word of tags: N, V, adj等
            
            D:          Number of observations. 即 fish, sleep等
            
            A:          The transition matrix. L x L
            
            O:          The observation matrix. L x D
            
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]


    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state 
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0. for _ in range(self.L)] for _ in range(M + 1)] # 各个路径下的probability
        seqs = [['' for _ in range(self.L)] for _ in range(M + 1)] # viterbi 算法实现的所有路径

        # L： # of states/tags, D: # of observations/总长度, M: sequence长度, A: L x L, O: L x D

        for i in range(self.L): # initialize 1st state, 根据题意 probs[1][i]为A_start * O, 而probs[0]为[0,...,0]
            probs[1][i] = self.A_start[i] * self.O[i][x[0]]
        
        for seq in range(1, M): # 遍历每个sequence
            for curr in range(self.L): # 遍历每个states/tags
                max_value = 0
                max_idx = 0
                for last in range(self.L): #遍历前一个state
                    if (probs[seq][last] * self.A[last][curr] * self.O[curr][x[seq]] >= max_value):
                        max_value = probs[seq][last] * self.A[last][curr] * self.O[curr][x[seq]] #记录下该路径及其概率
                        max_idx = last
                probs[seq + 1][curr] = max_value # 注意此处更新的是下一个state的probs
                seqs[seq + 1][curr] = max_idx
        
        max_seq_rev = []

        max_value = max(probs[M])
        max_idx = probs[M].index(max_value) # 寻找最后一个state中的max_value(即为max_prob)对应的max_idx
        
        max_seq_rev.append( str(max_idx) )

        for i in range(M, 1, -1):
            max_seq_rev.append( str(seqs[i][max_idx]) )
            max_idx = seqs[i][max_idx]
            
        max_seq = max_seq_rev[::-1]
        
        return "".join(max_seq)


    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        for i in range(self.L):
            alphas[1][i] = self.A_start[i] * self.O[i][x[0]]

        for seq in range(1, M): # 遍历每个sequence
            for curr in range(self.L): # 遍历每个states/tags
                sum_value = 0
                for last in range(self.L): #遍历前一个state
                    sum_value += alphas[seq][last] * self.A[last][curr] * self.O[curr][x[seq]] # Viterbi replaces sum with max

                alphas[seq + 1][curr] = sum_value

            if normalize:
                sum_alpha = sum(alphas[seq + 1])
                for curr in range(self.L):
                    alphas[seq + 1][curr] /= sum_alpha

        return alphas


    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of beta_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        for i in range(self.L):
            betas[-1][i] = 1 # PPT 74, beta(M) = 1

        for seq in range(-1, -M-1, -1): # 遍历每个sequence
            for curr in range(self.L): # 遍历每个states/tags
                sum_value = 0
                for nxt in range(self.L): #遍历后一个state
                    if seq != -M:
                        sum_value += betas[seq][nxt] * self.A[curr][nxt] * self.O[nxt][x[seq]] # PPT 74, 注意此处A_{z,j}变成A_{j,z}
                    else:
                        sum_value += betas[seq][nxt] * self.A_start[nxt] * self.O[nxt][x[seq]]

                betas[seq - 1][curr] = sum_value

            if normalize:
                sum_beta = sum(betas[seq - 1])
                for curr in range(self.L):
                    betas[seq - 1][curr] /= sum_beta


        return betas


    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''

        # Calculate each element of A using the M-step formulas.

        A_count = [[0. for i in range(self.L)] for j in range(self.L)]
        A_sum = [0. for i in range(self.L)]

        # For each input sequence:
        for y in Y: # Y is a list of lists of length L, so y should be a list, each y[i] is a mood tag
            for i in range(len(y) - 1): # A is calculated by 此状态到下一状态
                A_count[ y[i] ][ y[i + 1] ] += 1
                A_sum[y[i]] += 1
        
        for curr in range(self.L): # to normalize the A matrix
            for nxt in range(self.L):
                self.A[curr][nxt] = A_count[curr][nxt] / A_sum[curr]

        # Calculate each element of O using the M-step formulas.

        O_count = [[0. for i in range(self.D)] for j in range(self.L)] # O = L x D, (i, j) is the probability of emitting observation j given state i.
        O_sum = [0. for i in range(self.L)]

        for x, y in zip(X,Y):
            for i in range(len(y)):
                O_count[ y[i] ][ x[i] ] += 1
                O_sum[ y[i] ] += 1

        for curr in range(self.L):
            for nxt in range(self.D):
                self.O[curr][nxt] = O_count[curr][nxt] / O_sum[curr]

        pass


    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''

        for iteration in range(N_iters):

            A_count = [[0. for i in range(self.L)] for j in range(self.L)]
            A_sum = [0. for i in range(self.L)]
            O_count = [[0. for i in range(self.D)] for j in range(self.L)]
            O_sum = [0. for i in range(self.L)]

            # for each list
            for x in X:
                M = len(x)      # Length of sequence.
                # expectation step: given A & O matrix, predict probs of y’s for each training x
                # use forward-backward algorithm, 注意 alpha & beta 都是从 [1,...,M]
                alphas = self.forward(x, normalize=True) # (M+1)xL, (i, j)is alpha_j(i), probability of observing prefix x^1:i and state y^i = j.
                betas = self.backward(x, normalize=True)
                
                Marginals = [0. for _ in range(self.L)]
                for seq in range(1, M + 1): # for each prefix x^1:i
                    for curr in range(self.L): # 遍历每个states/tags
                        Marginals[curr] = alphas[seq][curr] * betas[seq][curr]
                    
                    Marginals_sum = sum(Marginals)
                    for curr in range(self.L):
                        Marginals[curr] /= Marginals_sum # normalized P(y^i (curr) | x), 可直接用于计算O
                        
                        # Maximization Step: Use y’s to estimate new (A,O)
                        O_count[curr][ x[seq - 1] ] += Marginals[curr] # seq从1开始, 而x[]从0开始
                        O_sum[curr] += Marginals[curr]
                        if seq != M: # A is calculated by 此状态到下一状态
                            A_sum[curr] += Marginals[curr] # for unsupervised learning, there is no tag, 无法更新 A_count

                # 需要计算 P(y^i, y^i+1 | x) 求A
                for seq in range(1, M):
                    A_update = [[0. for _ in range(self.L)] for _ in range(self.L)]

                    for curr in range(self.L):
                        for nxt in range(self.L):
                            A_update[curr][nxt] = alphas[seq][curr] * self.A[curr][nxt] * self.O[nxt][x[seq]] * betas[seq + 1][nxt]
                            # seq从1开始, 而x[]从0开始

                    A_update_sum = sum( [sum(A_update[i]) for i in range(len(A_update)) ] )

                    for curr in range(self.L):
                        for nxt in range(self.L):
                            A_update[curr][nxt] /= A_update_sum # normalized P(y^i, y^i+1 | x)

                    for curr in range(self.L):
                        for nxt in range(self.L):
                            A_count[curr][nxt] += A_update[curr][nxt]

            for curr in range(self.L): # to normalize the A & O matrix
                for nxt in range(self.L):
                    self.A[curr][nxt] = A_count[curr][nxt] / A_sum[curr]

            for curr in range(self.L):
                for nxt in range(self.D):
                    self.O[curr][nxt] = O_count[curr][nxt] / O_sum[curr]
        pass


    def get_data_with_distribute(self, dist): # 根据给定的概率分布随机返回数据（索引
        r = random.random()
        for i, p in enumerate(dist):
            if r < p:
                return i
            r -= p

    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        emission = []
        states = []
        
        y_start = random.randint(0, self.L -1) # starting state is chosen uniformly at random
        states.append(y_start)

        for i in range(M):
            
            # Generate observation/emission x
            #idx_random = random.randint(0, self.D -1)
            #max_value = max(self.O[ states[i] ])
            #x_index = self.O[ states[i] ].index(max_value)
            x_index = self.get_data_with_distribute(self.O[ states[i] ])
            emission.append(x_index)

            # Generate next state y.
            #idx_random = random.randint(0, self.L -1)
            y_index = self.get_data_with_distribute(self.A[ states[i] ])
            states.append(y_index)

        return emission, states[:-1]


    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)
    
    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states, N_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.
        
        N_iters:    The number of iterations to train on.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)
    
    # Compute L and D.
    L = n_states
    D = len(observations)

    
    # Randomly initialize and normalize matrix A.
    random.seed(2020)
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    random.seed(155)
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM
