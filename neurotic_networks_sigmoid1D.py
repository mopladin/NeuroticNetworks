import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import time
from matplotlib.collections import LineCollection

#####
def evaluate(w_t, x):

    x_t = np.append(x, -1)
    
    proj = np.dot(w_t, x_t)
    return 1 / (1 + np.exp(-proj))

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

    h = evaluate(w_t, x_s)

    x_s_t = np.append(x_s, -1)
    error = y_s - h
    delta_w = np.multiply(a * error, x_s_t)

    w_t = w_t + delta_w

    return w_t

#####
def test(w_t, X, y):
    
    N = X.shape[0]
    L = 1

    # Likelihood
    for i in range(N):

        x_i = X[i, :]
        y_i = y[i]

        h = evaluate(w_t, x_i)

        if y_i == 0:
            h = 1 - h

        L = L * h
    
    return L

#####
def visualize(X, y, w_t, axClass, cmap, i):

    axClass.clear()
    axClass.autoscale(False)

    # Plot samples
    axClass.scatter(X[:, 0], y[:], c=y, cmap=cmap, linewidths=0)

    # Plot sample chosen by stochastic gradient descent
    axClass.scatter(X[i, 0], y[i], s=40, c=[1 - y[i], 0, y[i]], cmap=cmap, linewidths=0)

    axClass.set_xlabel('Feature')
    axClass.set_ylabel('Label')

    # Plot sigmoid
    sig_x = np.arange(-1.5, 1.51, 0.01)
    sig_y = np.copy(sig_x)

    t = w_t[-1]
    w = w_t[0]

    for k in range(np.size(sig_x)):
        x_k = sig_x[k]
        sig_y[k] = evaluate(w_t, x_k)

    # Color sigmoid line segments according to sig_y
    points = np.array([sig_x, sig_y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=cmap,
                        norm=plt.Normalize(0, 1))

    lc.set_array(sig_y)
    lc.set_linewidth(3)
    axClass.add_collection(lc)

    plt.draw()


#####
def runEpoch(X, y, w_t, a, d, axClass, axLoss, cmap):

    N = X.shape[0]
    D = X.shape[1]

    idx = np.random.permutation(N)

    for i in idx:

        x_i = X[i, :]
        y_i = y[i]

        visualize(X, y, w_t, axClass, cmap, i)

        time.sleep(float(d) / 2)

        w_t = learnFromSample(x_i, y_i, w_t, a)

        visualize(X, y, w_t, axClass, cmap, i)

        time.sleep(d / 2)

    return w_t

#####
def initializeVisualization(X, y, cmap):

    plt.figure(figsize=(16,6))

    axClass = plt.subplot(121, autoscale_on=False, aspect='equal', adjustable='box-forced')
    axClass.set_xlabel('Feature')
    axClass.set_ylabel('Label')

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
def main():

    E = 100 # epoch count
    a = 1.0 # step size
    d = 1/1000 # delay in seconds

    cmap = LinearSegmentedColormap.from_list('mycmap', [(0, 'red'), 
                                                        (1, 'blue')]
                                            )

    ###
    X = np.array([[-1.0], [1.0]])
    y = np.array([0, 1])

    axClass, axLoss = initializeVisualization(X, y, cmap)

    w_t = np.array([1.0, 0])

    Likelihoods = []

    ###
    for e in range(E):
        w_t = runEpoch(X, y, w_t, a, d, axClass, axLoss, cmap)

        # Show likelihood
        L = test(w_t, X, y)
        Likelihoods.append(L)

        if e > 1:
            axLoss.plot([e - 1, e], [Likelihoods[e - 1], Likelihoods[e]], color='b')



    plt.show(block=True)

#####
main()
