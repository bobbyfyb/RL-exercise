import numpy as np 
import pandas as pd 
import time

N_STATES = 5 # assume user will ask 5 questions
ACTIONS = ['question', 'search']   # action space of igent 
EPSILON = 0.9  # ε-greedy's epsilon
ALPHA = 0.1 # learning rate
GAMMA = 0.9 # decay rate of the reward
MAX_EPISODES = 10 # max episodes

# bulid the q-learning table
def bulid_q_table(n_states, actions):
    table = pd.DataFrame(np.zeros((n_states, len(actions))),columns=actions,)
    return table

# action chosing using ε-greedy's method
def chose_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    if(np.random.uniform()>EPSILON) or (state_actions.all()==0):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.argmax()
    
    print(action_name)
    return action_name

# function when igent choose to question the user
def question(S, A):
    if S == N_STATES:
        print('user quit\n')
        S_ = 'qw'
        R = -1
    else:
        forward = np.random.randn() # assume igent will ask right or wrong question each with probability of 50%
        if forward <=0:
            print('wrong question\n')
            if S == 0:
                S_ = S
            else: 
                S_ = S - 1
            R = -1
        else:
            print('dialogue continue\n')
            S_ = S + 1
            R = 0
    return S_, R

# function when igent choose to retrieval the result. assume always return the right result.
def search(S, A):
    print('return search result\n')
    S_ = 'qr'
    R = 1
    return S_, R

def get_env_feedback(S, A):
    if A == 'question':
        S_, R = question(S, A)
    if A == 'search':
        S_, R = search(S, A)
    return S_, R


# main procedure of reinforcement learning 
def rl():
    q_table = bulid_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        while not is_terminated:
            print(str(S) + ':\n')

            # do q-learning
            A = chose_action(S, q_table)
            S_, R = get_env_feedback(S, A)
            q_predict = q_table.loc[S, A]
            if S_ != 'qr' and S_ != 'qw':
                q_target = R + GAMMA * q_table.iloc[S_, :].max() 
            else:
                q_target = R
                is_terminated = True
            q_table.loc[S, A]+=ALPHA*(q_target - q_predict)
            S = S_
            step_counter += 1
    return q_table


if __name__ == '__main__':
    q_table = rl()
    print('\r\nQ_table:\n')
    print(q_table)
