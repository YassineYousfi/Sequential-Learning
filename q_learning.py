import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

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
        
    print("Optimal policy is:\n", policy, "\nIts value is:\n", V[-1])
    rewards_smoothed = pd.Series(episode_rewards).rolling(int(num_episodes/10), min_periods=int(num_episodes/10)).mean()
    plt.plot(rewards_smoothed, c='red')
    plt.title("Reward cumulated over episodes smoothed over window size {}".format(int(num_episodes/10))) 
    plt.show()
    return V, policy, episode_rewards, N
