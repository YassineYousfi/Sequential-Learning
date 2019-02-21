import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from mdp_utils import samplefrom, playtransition

def epsilon_greedy_exploration(Q, epsilon, num_actions):
    def policy_exp(state):
        probs = np.ones(num_actions, dtype=float) * epsilon / num_actions
        best_action = np.argmax(Q[state])
        probs[best_action] += (1.0 - epsilon)
        return probs
    return policy_exp

def q_learning(mdp, num_episodes, T_max, epsilon=0.1):
    Q = np.zeros((mdp.S, mdp.A))
    episode_rewards = np.zeros(num_episodes)
    policy = np.ones(mdp.S)
    V = np.zeros((num_episodes, mdp.S))
    for i_episode in range(num_episodes): 
        greedy_probs = epsilon_greedy_exploration(Q, epsilon, mdp.A)
        N = np.zeros((mdp.S, mdp.A))
        state = 0
        for t in range(T_max):
            # epsilon greedy exploration
            action_probs = greedy_probs(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward = playtransition(mdp, state, action)
            episode_rewards[i_episode] += reward
            N[state, action] += 1
            alpha = 1/(t+1)**0.85
            best_next_action = np.argmax(Q[next_state])    
            td_target = reward + mdp.discount * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
            state = next_state
        V[i_episode,:] = Q.max(axis=1)
        policy = Q.argmax(axis=1)
        
    plot_summary(V, policy, episode_rewards, num_episodes)
    return V, policy, episode_rewards, N

def plot_summary(V, policy, episode_rewards, num_episodes):
    print("Optimal policy is:\n", policy, "\nIts value is:\n", V[-1])
    rewards_smoothed = pd.Series(episode_rewards).rolling(int(num_episodes/10), min_periods=int(num_episodes/10)).mean()
    plt.plot(rewards_smoothed, c='red')
    plt.title("Reward cumulated over episodes smoothed over window size {}".format(int(num_episodes/10))) 
    plt.show()

def UCB_exploration(Q, num_actions, alpha=1):
    def UCB_exp(state, N, t):
        A = np.zeros(num_actions)
        Q_ = Q[state,:]/max(Q[state,:]) + np.sqrt(alpha*np.log(t+1)/(2*N[state]))
        best_action = Q_.argmax()
        A[best_action] = 1
        return A
    return UCB_exp

def try_all_states_actions(mdp, N, Q):
    for state in range(mdp.S):
            for action in range(mdp.A):
                next_state, reward = playtransition(mdp, state, action)
                N[state, action] += 1
                best_next_action = np.argmax(Q[next_state])    
                td_target = reward + mdp.discount * Q[next_state][best_next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += 1 * td_delta
    return Q, N
    
def q_learning_ucb(mdp, num_episodes, T_max, epsilon=0.1):
    Q = np.zeros((mdp.S, mdp.A))
    episode_rewards = np.zeros(num_episodes)
    policy = np.ones(mdp.S)
    V = np.zeros((num_episodes, mdp.S))
    N = np.zeros((mdp.S, mdp.A))
    Q, N = try_all_states_actions(mdp, N, Q)
    for i_episode in range(num_episodes): 
        # UCB exploration
        UCB_exp = UCB_exploration(Q, mdp.A)
        state = np.random.choice(np.arange(mdp.S))
        for t in range(T_max):
            # UCB exploration
            action_probs = UCB_exp(state, N, t+(T_max*i_episode))
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward = playtransition(mdp, state, action)
            episode_rewards[i_episode] += reward
            N[state, action] += 1
            alpha = 1/(t+1)**0.8
            best_next_action = np.argmax(Q[next_state])    
            td_target = reward + mdp.discount * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
            state = next_state
        V[i_episode,:] = Q.max(axis=1)
        policy = Q.argmax(axis=1)
        
    plot_summary(V, policy, episode_rewards, num_episodes)
    return V, policy, episode_rewards, N