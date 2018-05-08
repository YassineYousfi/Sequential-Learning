import numpy as np
import matplotlib.pyplot as plt

def optimal_value_iteration(mdp, V0, num_iterations, epsilon=0.0001):
    V = np.zeros((num_iterations+1, mdp.S))
    V[0][:] = np.ones(mdp.S)*V0
    X = np.zeros((num_iterations+1, mdp.A, mdp.S))
    star = np.zeros((num_iterations+1,mdp.S))
    for k in range(num_iterations):
        for s in range(mdp.S):
            for a in range(mdp.A):
                X[k+1][a][s] = mdp.R[a][s] + mdp.discount*np.sum(mdp.P[a][s].dot(V[k]))
            star[k+1][s] = (np.argmax(X[k+1,:,s]))
            V[k+1][s] = np.max(X[k+1,:,s])
        
        if (np.max(V[k+1][:]-V[k][:])-np.min(V[k+1][:]-V[k][:]))<epsilon:
            V[k+1:][:]= V[k+1][:]
            star[k+1:][:]= star[k+1][:]
            X[k+1:][:][:]= X[k+1,:,:]
            break
        else: pass
    print("The optimal policy is:\n", star[-1], "\nIts value is:\n", V[-1])
    plt.plot(np.max(V,axis=1))
    plt.title("Evolution of ‖ V ‖∞ over iterations")
    plt.show()
    return star, V, X
