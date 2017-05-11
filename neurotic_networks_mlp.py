import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import time
from matplotlib.collections import LineCollection


#####
def evaluate(w_t, x):

    ###
    x_t = np.append(x, -1)

    hidden_weights = w_t[0:2, :]

    hidden_out = np.dot(hidden_weights, x_t)
    hidden_out = 1 / (1 + np.exp(-hidden_out))

    ###
    final_in = np.append(hidden_out, -1)
    final_weights = w_t[-1, :]

    proj = np.dot(final_weights, final_in)

    return 1 / (1 + np.exp(-proj)), final_in

#####
def classify(X, w_t):

    N = X.shape[0]

    t_append = -1 * np.ones((N, 1))

    X_t = np.append(X, t_append, 1)

    c_eval = np.dot(X_t, np.transpose(w_t))
    c_out = np.zeros(np.size(c_eval), dtype=int)

    c_out[np.where(c_eval > 0)] = 1

    return c_out

#####
def learnFromSample(x_s, y_s, w_t, a):

    #h, final_in = evaluate(w_t, x_s)

    #x_s_t = np.append(x_s, -1)
    #error = y_s - h
    #delta_w = np.multiply(a * error, final_in)

    # Update final layer
    #w_t[-1, :] = w_t[-1, :] + delta_w

    ###########
    ###
    x_t = np.append(x_s, -1)

    hidden_weights = w_t[0:2, :]

    hidden_out = np.dot(hidden_weights, x_t)
    hidden_out = 1 / (1 + np.exp(-hidden_out))

    ###
    final_in = np.append(hidden_out, -1)
    final_weights = w_t[-1, :]

    proj = np.dot(final_weights, final_in)

    final_out = 1 / (1 + np.exp(-proj))

    error = y_s - final_out
    delta_w_final = np.multiply(a * error, final_in)

    ### 
    w_t[-1, :] = w_t[-1, :] + delta_w_final

    ### Update hidden layer
    grads = np.multiply(hidden_out, (np.ones(np.shape(hidden_out)) - hidden_out))
    #grads = np.ones(np.shape(hidden_out))

    delta_w0 = np.multiply(a * error, np.multiply(grads[0], x_t))
    delta_w1 = np.multiply(a * error, np.multiply(grads[1], x_t))

    w_t[0, :] = w_t[0, :] + delta_w0
    w_t[1, :] = w_t[1, :] + delta_w1

    ##########

    return w_t

#####
def test(w_t, X, y):
    
    N = X.shape[0]
    L = 1

    # Likelihood
    for i in range(N):

        x_i = X[i, :]
        y_i = y[i]

        h,_ = evaluate(w_t, x_i)

        if y_i == 0:
            h = 1 - h

        L = L * h
    
    return L

#####
def visualize(X, y, w_t, axClass, cmap, i):

    axClass.clear()
    axClass.autoscale(False)

    # Plot activation
    res = 8.0
    h = np.zeros((res, res))

    for j in range(int(res)):
        for k in range(int(res)):
            x_k = np.array([(k - res / 2), (j - res / 2)]) * (3 / res)
            h[j, k], _ = evaluate(w_t, x_k)
            
    # Flip min and max for y so that image is flipped too
    axClass.imshow(h, extent=[-1.5, 1.5, 1.5, -1.5], cmap=cmap)

    ##########
    # Plot samples
    axClass.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, linewidths=1, zorder=2)

    # Plot sample chosen by stochastic gradient descent
    axClass.scatter(X[i, 0], X[i, 1], s=40, c=[1 - y[i], 0, y[i]], cmap=cmap, linewidths=1, zorder=2)

    axClass.set_xlabel('x')
    axClass.set_ylabel('y')

    plt.draw()


#####
def runEpoch(X, y, w_t, a, d, axClass, axLoss, cmap, e):

    N = X.shape[0]
    D = X.shape[1]

    idx = np.random.permutation(N)

    #for i in idx:
    for k in range(len(idx)):
        
        i = idx[k]

        x_i = X[i, :]
        y_i = y[i]

        visualize(X, y, w_t, axClass, cmap, i)
        #plt.savefig('e%d_k%d_0.png' % (e, k))

        time.sleep(float(d) / 2)

        w_t = learnFromSample(x_i, y_i, w_t, a)

        visualize(X, y, w_t, axClass, cmap, i)
        #plt.savefig('e%d_k%d_1.png' % (e, k))

        time.sleep(d / 2)

    return w_t

#####
def initializeVisualization(X, y, cmap):

    plt.figure(figsize=(16,6))

    axClass = plt.subplot(121, autoscale_on=False, aspect='equal', adjustable='box-forced')
    axClass.set_xlabel('x')
    axClass.set_ylabel('y')

    axLoss = plt.subplot(122)
    axLoss.grid(True)
    axLoss.set_xlabel('Epoch')
    axLoss.set_ylabel('Likelihood')

    v = 1.5
    axClass.axis([-v, v, -v, v])

    plt.draw()
    plt.show(block=False)

    return axClass, axLoss

#####
def generateData(N, D):

    #X = 1 - 2 * np.random.rand(N, D)
    #y = np.random.randint(0, 2, N)

    size_a = 1.5    # cluster-to-cluster
    size_b = 1.25   # cluster-internal

    shifts = 1 - size_a * np.random.rand(1, D)
    shifts = np.append(shifts, -shifts, 0)


    X = size_b * np.add(np.random.rand(N / 2, D), shifts[0, :])
    X = np.append(X, size_b * np.add(np.random.rand(N / 2, D), shifts[1, :]), 0)

    stretch = np.random.rand(1, D)
    stretch = stretch / np.linalg.norm(stretch)
    X = np.multiply(X, stretch)

    y = np.ones(N / 2)
    y = np.append(y, np.zeros(N / 2))

    # Scale to have values of at most 1.25 to fit the viewport
    x_max = np.maximum(np.amax(X), -np.amin(X))
    X = X * 1.25 / x_max

    return (X, y)

#####
def main():

    N = 10
    D = 2
    E = 500 # epoch count
    a = 2.5 # step size
    d = 1/1000 # delay in seconds

    cmap = LinearSegmentedColormap.from_list('mycmap', [(0, 'red'), 
                                                        (1, 'blue')]
                                            )

    ###
    X = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [1.0, -1.0]])
    y = np.array([0, 1, 0, 1])
    #X, y = generateData(N, D)

    axClass, axLoss = initializeVisualization(X, y, cmap)

    #w_t = np.array([[-2.0, 2.0, 2.0], [2.0, -2.0, 2.0], [1.0, 0.0, 0.0]])
    w_t = 1 - 2*np.random.rand(3, D)
    w_t = np.append(w_t, np.zeros((3, 1)), 1)

    #print w_t

    Likelihoods = []

    ###
    for e in range(E):
        w_t = runEpoch(X, y, w_t, a, d, axClass, axLoss, cmap, e)

        # Show likelihood
        L = test(w_t, X, y)
        Likelihoods.append(L)

        if e > 0:
            axLoss.plot([e, e + 1], [Likelihoods[e - 1], Likelihoods[e]], color='b')

    plt.show(block=True)

#####
main()
