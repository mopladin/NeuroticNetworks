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

    # Plot class line
    w_normed = w_t[:-1] / np.linalg.norm(w_t[0:-1])
    diag = np.linalg.norm([1.5, 1.5])

    sig_x = w_normed[0] * np.arange(-diag, diag, 0.01)
    sig_y = w_normed[1] * np.arange(-diag, diag, 0.01)

    sig_h = np.copy(sig_x)

    for k in range(np.size(sig_h)):
        x_k = [sig_x[k], sig_y[k]]
        sig_h[k] = evaluate(w_t, x_k)

    points = np.array([sig_x, sig_y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=cmap,
                        norm=plt.Normalize(0, 1))

    lc.set_array(sig_h)
    lc.set_linewidth(999)
    axClass.add_collection(lc)


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
def main():

    E = 100 # epoch count
    a = 2.5 # step size
    d = 1/1000 # delay in seconds

    cmap = LinearSegmentedColormap.from_list('mycmap', [(0, 'red'), 
                                                        (1, 'blue')]
                                            )

    ###
    X = np.array([[-0.5, 1], [1, 0], [0.5, -1], [1, 1], [0, 1], [1, -1]])
    y = np.array([0, 1, 0, 1, 0, 1])

    axClass, axLoss = initializeVisualization(X, y, cmap)

    w_t = np.array([1.0, 0.0, 0])

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
