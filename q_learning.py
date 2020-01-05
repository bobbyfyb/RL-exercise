import numpy as np 
import pandas as pd 
import time

N_STATES = 10 # width of the 1-dim world
ACTIONS = ['left', 'right'] # action space
EPSILON = 0.9 # ε-greedy's epsilon
ALPHA = 0.1 # learning rate
GAMMA = 0.9 # decay rate of the reward
MAX_EPISODES = 15 # max episodes
FRESH_TIME = 0.01 # speed 

def bulid_q_table(n_states, actions):
    table = pd.DataFrame(np.zeros((n_states, len(actions))),columns=actions,)
    return table

def chose_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    if(np.random.uniform()>EPSILON) or (state_actions.all()==0):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.argmax()
    return action_name

def get_env_feedback(S, A):
    if A == 'right':
        if S == N_STATES - 2:
            S_= 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S-1
    return S_, R

def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

def rl():
    q_table = bulid_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:

            A = chose_action(S, q_table)
            S_, R = get_env_feedback(S, A)
            q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()
            else:
                q_target = R
                is_terminated = True
        
            q_table.loc[S, A]+=ALPHA*(q_target - q_predict)
            S = S_

            update_env(S, episode, step_counter+1)

            step_counter += 1
    return q_table


if __name__ == '__main__':
    q_table = rl()
    print('\r\nQ_table:\n')
    print(q_table)