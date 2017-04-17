import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import time


def debugPrintArray(ar, name):
    print "%s: \n%s" % (name, np.array_str(ar))


def generateData(N, D):

    X = 1 - 2 * np.random.rand(N, D)
    y = np.random.randint(0, 2, N)

    return (X, y)


def initializeWeights(D):

    w = np.random.randn(D)
    w = w / np.linalg.norm(w)

    t = 0

    w_t = np.append(w, t)

    return w_t


def classify(X, w_t):

    N = X.shape[0]

    t_append = -1 * np.ones((N, 1))

    X_t = np.append(X, t_append, 1)

    c_eval = np.dot(X_t, np.transpose(w_t))
    c_out = np.zeros(np.size(c_eval), dtype=int)

    c_out[np.where(c_eval > 0)] = 1

    return c_out


def initializeVisualization(X, y, cmap):

    plt.figure(figsize=(15,8))

    axScatter = plt.subplot(121, autoscale_on=False, aspect='equal', adjustable='box-forced')
    axClass = plt.subplot(122, autoscale_on=False, aspect='equal', adjustable='box-forced', sharex=axScatter, sharey=axScatter)

    axScatter.title.set_text('Ground truth')

    axScatter.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, linewidths=0)

    #viewport
    v = 1.5
    axScatter.axis([-v, v, -v, v])
    axClass.axis([-v, v, -v, v])

    plt.draw()
    plt.show(block=False)

    return (axScatter, axClass)


def visualize(X, y, w_t, c, E, axClass, cmap, k):

    #print "x: \n%s" % np.array_str(x)
    #print "w: \n%s" % np.array_str(w)
    #print "c_eval: \n%s" % np.array_str(c_eval)
    #print "c_out: \n%s" % np.array_str(c_out)
    #print "c_gt: \n%s" % np.array_str(c_gt)

    ##### Visualize
    axClass.clear()

    axClass.autoscale(False)

    #c = classify(X, w_t)

    #debugPrintArray(c, "c")
    #debugPrintArray(y, "y")

    
    acc = float(np.sum(y == c)) / float(len(c))

    # Samples
    title = "Classification, E: %s, Acc: %f" % (E+1, acc)
    axClass.title.set_text(title)
    axClass.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, linewidths=0)

    axClass.scatter(X[k, 0], X[k, 1], s=40, c=[1 - y[k], 0, y[k]], cmap=cmap, linewidths=0)
    #plt.cm.get_cmap('jet')

    # Class line normal
    t = w_t[-1]
    w = w_t[0:-1]

    #print "t: \n%s" % np.array_str(t)
    #print "w: \n%s" % np.array_str(w)

    norm_start = w * t
    norm_end = norm_start + w * 0.1

    # Class line
    axClass.plot([norm_start[0], norm_end[0]], [norm_start[1], norm_end[1]])

    w_ortho = [w[1], -w[0]]
    cl_start = norm_start - w_ortho
    cl_end = norm_start + w_ortho

    axClass.plot([cl_start[0], cl_end[0]], [cl_start[1], cl_end[1]])

    plt.draw()


def learnFromSample(x_s, y_s, w_t, a):

    c = classify(x_s, w_t)

    x_s_t = np.array([x_s[0,0], x_s[0,1], -1])

    if c != y_s:

        delta_w = np.multiply(a * (y_s - c), x_s_t)

        w_t = w_t + delta_w

        w = w_t[0:2]
        w = w / np.linalg.norm(w)

        w_t[0:2] = w[0:2]


    return w_t


def runEpoch(X, y, w_t, a, E, d, axClass, cmap):

    N = X.shape[0]
    D = X.shape[1]

    idx = np.random.permutation(N)

    for k in idx:

        x_s = np.reshape(X[k, :], (1, D))
        y_s = y[k]

        c = classify(X, w_t)

        #visualize(X, y, w_t, c, E, axClass, cmap, k)
        axClass.scatter(X[k, 0], X[k, 1], s=40, c=[1 - y[k], 0, y[k]], cmap=cmap, linewidths=0)
        time.sleep(d / 2)

        w_t = learnFromSample(x_s, y_s, w_t, a)

        c = classify(X, w_t)
            
        visualize(X, y, w_t, c, E, axClass, cmap, k)

        if np.array_equal(c, y):
            return (w_t, True)


        time.sleep(d / 2)

    return (w_t, False)


def main():

    N = 5 # sample count
    D = 2 # feature dimension count

    E = 50 # epoch count
    a = 0.5 # step size
    e = 0.95 # epoch learn factor
    d = 1 / 100 # delay in seconds

    cmap = LinearSegmentedColormap.from_list('mycmap', [(0, 'red'), 
                                                        (1, 'blue')]
                                            )

    ###
    X, y = generateData(N, D)
    axScatter, axClass = initializeVisualization(X, y, cmap)

    w_t = initializeWeights(D)

    ###
    for k in range(E):
        w_t, finished = runEpoch(X, y, w_t, a, k, d, axClass, cmap)
        a = a * e

        if finished:
            break
            

    plt.show(block=True)

main()
