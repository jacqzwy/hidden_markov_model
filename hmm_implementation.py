import numpy as np
import os
from timeit import default_timer as timer

def forward(O, A, B, pi, q_num):
    '''Forward algorithm
    Parameters:
        O (numpy array of integers): observed sequence 
        A (numpy matrix of floats): transition probability matrix
        B (numpy matrix of floats): emission probability matrix
        pi (numpy array of floats): initial state distribution
        q_num (int): number of states
        
    Returns:
        alpha (numpy matrix): alpha values for states 1:M, times 1:T 
    '''
    T = len(O)
    #create alpha matrix
    alpha = np.zeros(shape=(q_num,T))
    
    #initialization for T=1
    for q in range(q_num):
        alpha[q, 0] = pi[q] * B[q,O[0]]
        
    #recursive steps for T = 2:T
    for t in range(1,T):
        for q in range(q_num):
             probs_to_sum = np.zeros(q_num)
             for i in range(q_num):
                 probs_to_sum[i] = alpha[i,t-1] * A[i,q]
             alpha[q,t] = (np.sum(probs_to_sum)
                           * B[q, O[t]])
    return alpha

def backward(O, A, B, pi, q_num):
    '''Backward algorithm
    Parameters:
        O (numpy array of integers): observed sequence 
        A (numpy matrix of floats): transition probability matrix
        B (numpy matrix of floats): emission probability matrix
        pi (numpy array of floats): initial state distribution
        q_num (int): number of states
        
    Returns:
        beta (numpy matrix of floats): beta values for states
        1:M, times 1:T 
    '''
    T = len(O)
    #create beta matrix
    beta = np.zeros(shape=(q_num, T))
    
    #initialization for T=T
    for q in range(q_num):
        beta[q, T-1] = 1
        
    #recursive steps for T-1:1
    for t in range(T-2, -1, -1):
        for q in range(q_num):
            probs_to_sum = np.zeros(q_num)
            for j in range(q_num):
                probs_to_sum[j] = (A[q, j]
                                   * B[j, O[t+1]]
                                   * beta[j, t+1])
            beta[q, t] = np.sum(probs_to_sum) 
    return beta
    
def baum_welch(train_data, A, B, pi, q_num, n_iters):
    ''' Baum-Welch algorithm
    Parameters:
        train (numpy array of arrays): observed sequence(s) 
        A (numpy matrix of floats): transition probability matrix
        B (numpy matrix of floats): emission probability matrix
        pi (numpy array of floats): initial state distribution
        q_num (int): number of states
        
    Dependencies:
        forward(O, A, B, pi, q_num)
        backward(O, A, B, pi, q_num)
        
    Returns:
        (A, B, pi) (tuple) where:
        A (numpy matrix): transition probability matrix
        B (numpy matrix): emission probability matrix
        pi (numpy array of floats): initial state distribution
    '''
    A = np.copy(A)
    B = np.copy(B)
    pi = np.copy(pi)
    
    for n in range(n_iters):
        print('Iter', n+1)
        #create matrices for A, B, pi, array to store P(O) 
        A1 = np.zeros_like(A)
        B1 = np.zeros_like(B)
        pi1 = np.zeros_like(pi)
        prob_O = np.zeros(len(train_data))
        
        #create xi and gamma matrices
        T = len(train_data[0])  
        K = len(train_data)
        xi = np.zeros(shape=(q_num, q_num, T-1,K))
        gamma = np.zeros(shape=(q_num,T,K))
        
        #use training data to estimate A, B, pi 
        for k, O in enumerate(train_data): 
            alpha = forward(O, A, B, pi, q_num)
            beta = backward(O, A, B, pi, q_num)
            prob_O[k] = np.sum(alpha[:,-1])

            #calculate xi matrix
            for i in range(q_num):
                for j in range(q_num):
                    for t in range(T-1):
                        xi[i,j,t,k] = (alpha[i,t] * A[i,j] 
                                     * B[j, O[t+1]] * beta[j,t+1]) / prob_O[k]
            
            #calculate gamma matrix
            gamma_k = np.sum(xi[:,:,:,k], axis=1)
            gamma_kt = (alpha[:,-1]/prob_O[k]).reshape((q_num,1))
            gamma_k = np.hstack((gamma_k, gamma_kt))            
            gamma[:,:,k] = gamma_k  
            
            
        #estimate A, B, pi after all xi and gamma calculated for all sequences
        #calculate overall gamma and xi matrices
        gamma_all = np.sum(gamma, axis=2)
        xi_all = np.sum(xi, axis=2)
        
        #estimate pi
        pi1 = gamma_all[:,0]/ K
        
        #estimate B
        V = B.shape[1] #num of symbols observed
        B_denom_v = np.sum(gamma_all, axis=1).reshape((-1,1)).ravel()
        B_denom = np.repeat(B_denom_v[:,np.newaxis], V, 1)

        B_numer = np.zeros(shape=(q_num,V))
        for k, O in enumerate(train_data): #for each seq
            for v in range(V): #for each symbol
                    col_to_sum = np.argwhere(O==v)
                    gamma_to_sum = np.sum(gamma[:, col_to_sum, k], axis=1).ravel()
                    B_numer[:,v] += gamma_to_sum
        
        B1 = B_numer/ B_denom

        #estimate A
        A_denom = np.sum(gamma_all[:,:-1], axis=1).reshape((-1,1))
        A_numer = np.sum(xi_all, axis=2)        
        A1 = np.divide(A_numer, A_denom)
         
        #check if likelihood of new is close to likelihood of old
        #calculate likelihood of old
        l0 = np.sum(np.log(prob_O))
        #calculate likelihood of new
        prob_O1 = np.zeros_like(prob_O)
        for k, O in enumerate(train_data):       
            alpha = forward(O, A1, B1, pi1, q_num)
            prob_O1[k] = np.sum(alpha[:,-1])

        l1 = np.sum(np.log(prob_O1))

        if l1 - l0 < 10**(-3):        
            print(l0)
            print(l1)
            print('number of iterations', n+1)
            return (A, B, pi)
        
        A = A1
        B = B1
        pi = pi1
            
    return (A, B, pi)

def calc_likelihood(train_data, A, B, pi, q_num):
    ''' Calculate likelihood of multiple sequences
    Parameters:
        train (numpy array of arrays): observed sequence(s) 
        A (numpy matrix of floats): transition probability matrix
        B (numpy matrix of floats): emission probability matrix
        pi (numpy array of floats): initial state distribution
        q_num (int): number of states
        
    Dependencies:
        forward(O, A, B, pi, q_num)
        
    Returns:
        loglikelihood of sequences
    ''' 
    prob_O = np.zeros(len(train_data))
    for k, O in enumerate(train_data): 
        alpha = forward(O, A, B, pi, q_num)
        prob_O[k] = np.sum(alpha[:,-1])
        loglhd = np.sum(np.log(prob_O))
    return loglhd
    

def viterbi(O, A, B, pi, q_num):
    ''' Viterbi algorithm
    Parameters:
        O (numpy array of integers): observed sequence 
        A (numpy matrix of floats): transition probability matrix
        B (numpy matrix of floats): emission probability matrix
        pi (numpy array of floats): initial state distribution
        q_num (int): number of states
        
    Returns:
        q_opt (numpy array): Most probable state sequence
    '''
    T = len(O)
    
    #create probabilities and pointer tables
    probs = np.zeros(shape=(q_num,T))
    point = np.zeros(shape=(q_num,T), dtype = int)
    
    #initialization for T=1
    #pointer(q,0) = 0 already defined
    for q in range(q_num):
        probs[q,0] = pi[q] * B[q, O[0]]
    
    #recursive steps for 2:T
    for t in range(1,T):
        for q in range(q_num):
            
            #fill probabilities table
            prev_probs = [probs[i, t-1] * A[i,q]  
                            for i in range(q_num)]
            probs[q,t] = np.max(prev_probs) * B[q, O[t]]    
            #fill pointer table
            point[q,t] = np.argmax(prev_probs)
    
    #backtrack through pointers
    q_opt = np.zeros(T, dtype = int)
    #find last state
    last_q = np.argmax([probs[q, T-1] for q in range(q_num)]) 
    q_opt[-1] = last_q
    
    #backtrack
    for t in range(T-1, 0, -1):
        q_opt[t-1] = point[last_q, t]
        last_q = q_opt[t-1]
    
    return q_opt

def predict(O, A, B, pi, q_num):
    '''Predict next hidden state and observation
    Parameters:
        O (numpy array of integers): observed sequence 
        A (numpy matrix of floats): transition probability matrix
        B (numpy matrix of floats): emission probability matrix
        pi (numpy array of floats): initial state distribution
        q_num (int): number of states
        
    Returns:
        next_q (int): next hidden state
        next_O (int): next observed symbol
    '''
    most_likely_seq = viterbi(O, A, B, pi, q_num)
    last_q = most_likely_seq[-1] 
    next_q = np.where(A[last_q,:] == np.max(A[last_q,:]))[0][0]
    next_O = np.where(B[next_q,:] == np.max(B[next_q,:]))[0][0]
    return next_q, next_O

#training data
#6 states, sample of 500 obs
filename = 'train534.dat'
with open(filename, 'r') as file:
    read_strings = file.read().splitlines()
    
seq_str = [x.split() for x in read_strings]
seq = [[int(x) for x in s] for s in seq_str]    

#A
np.random.seed(743)
A = np.random.rand(6,6)
A = A/A.sum(axis=1)[:,None]
#print('A')
#print(A)

#B
B = np.random.rand(6,4)
B = B/B.sum(axis=1)[:,None]
#print('B')
#print(B)

# reorder states according to criteria given
def largest_indices(mat, n):
    flat = mat.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, mat.shape)

row, col = largest_indices(B, 24)
#print(row)
_, idx = np.unique(row, return_index=True)
#print(row[np.sort(idx)])

Bsort = np.stack((B[5,:], B[3,:], B[2,:], B[4,:], B[1,:], B[0,:]), axis = 0)
#print(Bsort)

#pi
pi = np.random.rand(1,6)
pi1 = pi/pi.sum(axis=1)[:,None]
pi = pi1.ravel()

#print('pi')
#print(pi)

#take 500 samples
seq = np.array(seq)
seq_train = seq[np.random.randint(0,1000,size=500), :]

start = timer()
A_est, B_est, pi_est = baum_welch(seq_train, A, Bsort, pi, 6, 2000)
end= timer()
#print(end-start)
#
#print('A')
#print(A_est)
#print('B')
#print(B_est)
#print('pi')
#print(pi_est)

#run Viterbi on all sequences
start_v = timer()
hidden_states = np.zeros(shape=(len(seq),len(seq[0])), dtype=int)
for i, O in enumerate(seq):
    hidden_states[i,:] = viterbi(O, A_est, B_est, pi, 6)
end_v = timer()
#print(end_v - start_v)
    
#test data
filename1 = 'test1_534.dat'
with open(filename1, 'r') as file1:
    read_strings1 = file1.read().splitlines()

test_str = [x.split() for x in read_strings1]
test = [[int(x) for x in s] for s in test_str] 
test = np.array(test)

#calculate complete likelihood
likelihood = calc_likelihood(test, A_est, B_est, pi_est, 6)

#run Viterbi on test data
test_hidden_states = np.zeros(shape=(len(test),len(test[0])), dtype=int)
for i, O in enumerate(test):
    test_hidden_states[i,:] = viterbi(O, A_est, B_est, pi_est, 6)

#predict next hidden state and output
predictions = []
for i, O in enumerate(test):
    predictions.append(predict(O, A_est, B_est, pi_est, 6))
predicted_output = np.array([x[1] for x in predictions]).reshape(-1,1)
