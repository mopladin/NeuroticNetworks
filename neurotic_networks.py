import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt


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
    c_out = np.zeros(np.size(c_eval))

    c_out[np.where(c_eval > 0)] = 1

    return c_out


def visualize(X, y, c, w_t):


    #cmap = plt.cm.get_cmap('jet')

    #print "x: \n%s" % np.array_str(x)
    #print "w: \n%s" % np.array_str(w)
    #print "c_eval: \n%s" % np.array_str(c_eval)
    #print "c_out: \n%s" % np.array_str(c_out)
    #print "c_gt: \n%s" % np.array_str(c_gt)

    ##### Visualize
    plt.figure(figsize=(15,8))
    axScatter = plt.subplot(121, autoscale_on=False, aspect='equal', adjustable='box-forced')
    axClass = plt.subplot(122, autoscale_on=False, aspect='equal', adjustable='box-forced', sharex=axScatter, sharey=axScatter)

    cmap = LinearSegmentedColormap.from_list('mycmap', [(0, 'red'), 
                                                        (1, 'blue')]
                                            )
    # Samples
    axScatter.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, linewidths=0)
    axClass.scatter(X[:, 0], X[:, 1], c=c, cmap=cmap, linewidths=0)

    axScatter.title.set_text('Ground truth')
    axClass.title.set_text('Classification')

    #viewport
    v = 1.5
    axScatter.axis([-v, v, -v, v])
    axClass.axis([-v, v, -v, v])

    # Class line normal
    t = w_t[-1]
    w = w_t[0:-1]

    #print "t: \n%s" % np.array_str(t)
    #print "w: \n%s" % np.array_str(w)

    norm_start = w * t
    norm_end = norm_start + w * 0.1

    # Class line normal (TODO: Add threshold)
    axClass.plot([norm_start[0], norm_end[0]], [norm_start[1], norm_end[1]])

    w_ortho = [w[1], -w[0]]
    cl_start = norm_start - w_ortho
    cl_end = norm_start + w_ortho

    print w_ortho

    axClass.plot([cl_start[0], cl_end[0]], [cl_start[1], cl_end[1]])



    # Hint to class line
    plt.draw()
    plt.show()


def main():

    N = 5
    D = 2

    X, y = generateData(N, D)
    w_t = initializeWeights(D)
    c = classify(X, w_t)

    visualize(X, y, c, w_t)


main()
