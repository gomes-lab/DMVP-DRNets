import numpy as np
import pickle
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.manifold import TSNE

eps = 1e-9

def tSNE(X):
    import numpy as Math
    import pylab as Plot
    
    def Hbeta(D = Math.array([]), beta = 1.0):
        """Compute the perplexity and the P-row for a specific value of the precision of a Gaussian distribution."""
    
        # Compute P-row and corresponding perplexity
        #print (np.max(-D.copy() * beta), np.min(-D.copy() * beta))
        P = Math.exp(-D.copy() * beta);
        sumP = sum(P + eps);
        H = Math.log(sumP) + beta * Math.sum(D * P) / sumP;
        P = P / sumP;
        return H, P;
    
    
    def x2p(X = Math.array([]), tol = 1e-5, perplexity = 20.0):
        """Performs a binary search to get P-values in such a way that each conditional Gaussian has the same perplexity."""
    
        # Initialize some variables
        print("Computing pairwise distances...")
        (n, d) = X.shape;
        sum_X = Math.sum(Math.square(X), 1);

        #D =  -Math.dot(X, X.T) #   negative inner product as the distance 
        D = Math.add(Math.add(-2 * Math.dot(X, X.T), sum_X).T, sum_X) # L2 distance


        P = Math.zeros((n, n));
        beta = Math.ones((n, 1));
        logU = Math.log(perplexity);
    
        # Loop over all datapoints
        for i in range(n):
    
            # Print progress
            if i % 500 == 0:
                print("Computing P-values for point ", i, " of ", n, "...")
    
            # Compute the Gaussian kernel and entropy for the current precision
            betamin = -Math.inf;
            betamax =  Math.inf;
            Di = D[i, Math.concatenate((Math.r_[0:i], Math.r_[i+1:n]))];
            (H, thisP) = Hbeta(Di, beta[i]);
    
            # Evaluate whether the perplexity is within tolerance
            Hdiff = H - logU;
            tries = 0;
            while Math.abs(Hdiff) > tol and tries < 50:
    
                # If not, increase or decrease precision
                if Hdiff > 0:
                    betamin = beta[i].copy();
                    if betamax == Math.inf or betamax == -Math.inf:
                        beta[i] = beta[i] * 2;
                    else:
                        beta[i] = (beta[i] + betamax) / 2;
                else:
                    betamax = beta[i].copy();
                    if betamin == Math.inf or betamin == -Math.inf:
                        beta[i] = beta[i] / 2;
                    else:
                        beta[i] = (beta[i] + betamin) / 2;
    
                # Recompute the values
                (H, thisP) = Hbeta(Di, beta[i]);
                Hdiff = H - logU;
                tries = tries + 1;
    
            # Set the final row of P
            P[i, Math.concatenate((Math.r_[0:i], Math.r_[i+1:n]))] = thisP;
    
        # Return final P-matrix
        print("Mean value of sigma: ", Math.mean(Math.sqrt(1 / beta)));
        return P;
    
    
    def pca(X = Math.array([]), no_dims = 50):
        """Runs PCA on the NxD array X in order to reduce its dimensionality to no_dims dimensions."""
    
        print("Preprocessing the data using PCA...")
        (n, d) = X.shape;
        X = X - Math.tile(Math.mean(X, 0), (n, 1));
        (l, M) = Math.linalg.eig(Math.dot(X.T, X));
        print("singular values:\n", l)
        Y = Math.dot(X, M[:,0:no_dims]);
        return Y;
    
    
    def tsne(X = Math.array([]), no_dims = 2, initial_dims = 50, perplexity = 40.0, max_iter=1000):
        """Runs t-SNE on the dataset in the NxD array X to reduce its dimensionality to no_dims dimensions.
        The syntaxis of the function is Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array."""
    
        # Check inputs
        if isinstance(no_dims, float):
            print("Error: array X should have type float.")
            return -1;
        if round(no_dims) != no_dims:
            print("Error: number of dimensions should be an integer.")
            return -1;
    
        # Initialize variables
        X = pca(X, initial_dims).real;
        (n, d) = X.shape;
        initial_momentum = 0.5;
        final_momentum = 0.8;
        eta = 50;
        min_gain = 0.01;
        Y = Math.random.randn(n, no_dims);
        dY = Math.zeros((n, no_dims));
        iY = Math.zeros((n, no_dims));
        gains = Math.ones((n, no_dims));
    
        # Compute P-values
        P = x2p(X, 1e-5, perplexity);
        P = P + Math.transpose(P);
        P = P / Math.sum(P);
        P = P * 4;                                    # early exaggeration
        P = Math.maximum(P, 1e-12);
    
        # Run iterations
        for iter in range(max_iter):
    
            # Compute pairwise affinities
            sum_Y = Math.sum(Math.square(Y), 1);
            num = 1 / (1 + Math.add(Math.add(-2 * Math.dot(Y, Y.T), sum_Y).T, sum_Y));
            num[range(n), range(n)] = 0;
            Q = num / Math.sum(num);
            Q = Math.maximum(Q, 1e-12);
    
            # Compute gradient
            PQ = P - Q;
            for i in range(n):
                dY[i,:] = Math.sum(Math.tile(PQ[:,i] * num[:,i], (no_dims, 1)).T * (Y[i,:] - Y), 0);
    
            # Perform the update
            if iter < 20:
                momentum = initial_momentum
            else:
                momentum = final_momentum
            gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0));
            gains[gains < min_gain] = min_gain;
            iY = momentum * iY - eta * (gains * dY);
            Y = Y + iY;
            Y = Y - Math.tile(Math.mean(Y, 0), (n, 1));
    
            # Compute current value of cost function
            if (iter + 1) % 10 == 0:
                C = Math.sum(P * Math.log(P / Q));
                print("Iteration ", (iter + 1), ": error is ", C)
    
            # Stop lying about P-values
            if iter == 100:
                P = P / 4;
    
        # Return solution
        return Y;

    return tsne(X, 2, 80, 30, 2000) 

def correlation(cov, show=False):
    indices = []
    sigma = np.zeros((89,89))
    for i in range(89):
        for j in range(89):
            sigma[i][j]=cov[i][j]/np.sqrt(cov[i][i]*cov[j][j])
            indices.append((i,j))
    plt.imshow(sigma, cmap='jet', interpolation='nearest')
    if (show):
        plt.show()
    return sigma, indices

def show_embed(L, ele_dict):
    plt.clf()
    #X_embedded = TSNE(n_components=2, init='pca', perplexity = 8.0).fit_transform(L) 
    X_embedded = tSNE(L)
    x_min, x_max = X_embedded.min(0), X_embedded.max(0)
    X_embedded = (X_embedded - x_min) / (x_max - x_min)
    #ele_color = np.load("ele_color.npy").item()
    print (X_embedded.shape)

    for i in range(X_embedded.shape[0]):
        x = X_embedded[i][0]
        y = X_embedded[i][1]
        #print(x, y, ele_dict[i])
        plt.text(x, y, ele_dict[i], fontsize=8, ha='center', va='center')
        key = str(ele_dict[i])
        #print(key)
        #plt.plot(x, y, 'o', ms=20, c=ele_color[key])
        plt.plot(x, y, 'o', ms=10)
    plt.savefig("./small_feature_emb.jpg")
    #plt.savefig("./cor_emb.jpg")
    #plt.show()

def show_embed_correlation(L, ele_dict, mon):
    plt.clf()
    #X_embedded = TSNE(n_components=2, init='pca', perplexity = 8.0).fit_transform(L) 
    X_embedded = tSNE(L)
    x_min, x_max = X_embedded.min(0), X_embedded.max(0)
    X_embedded = (X_embedded - x_min) / (x_max - x_min)
    #ele_color = np.load("ele_color.npy").item()
    print (X_embedded.shape)

    for i in range(X_embedded.shape[0]):
        x = X_embedded[i][0]
        y = X_embedded[i][1]
        #print(x, y, ele_dict[i])
        plt.text(x, y, ele_dict[i], fontsize=8, ha='center', va='center')
        key = str(ele_dict[i])
        #print(key)
        #plt.plot(x, y, 'o', ms=20, c=ele_color[key])
        plt.plot(x, y, 'o', ms=10)
    plt.savefig("./vis_%d/cor_emb.jpg" % (mon))
    #plt.show()

def show_embed_feature(L, ele_dict, mon):
    plt.clf()
    #X_embedded = TSNE(n_components=2, init='pca', perplexity = 8.0).fit_transform(L) 
    X_embedded = tSNE(L)
    x_min, x_max = X_embedded.min(0), X_embedded.max(0)
    X_embedded = (X_embedded - x_min) / (x_max - x_min)
    #ele_color = np.load("ele_color.npy").item()
    print (X_embedded.shape)

    for i in range(X_embedded.shape[0]):
        x = X_embedded[i][0]
        y = X_embedded[i][1]
        #print(x, y, ele_dict[i])
        plt.text(x, y, ele_dict[i], fontsize=8, ha='center', va='center')
        key = str(ele_dict[i])
        #print(key)
        #plt.plot(x, y, 'o', ms=20, c=ele_color[key])
        plt.plot(x, y, 'o', ms=10)
    plt.savefig("./vis_%d/feature_emb.jpg" % (mon))
    #plt.savefig("./cor_emb.jpg")
    #plt.show()



mons = [i for i in range(1, 13)]

#for mon in mons:

parser = argparse.ArgumentParser()
parser.add_argument('-mon', "--mon", default=1, type=int)
args = parser.parse_args()

for mon in mons:
    #mon = args.mon
    cov = np.load("./data/vis_%d/cov.npy" % (mon))
    correlation_emb = np.linalg.cholesky(cov)
    feature_emb = np.load("vis_%d/feature_emb_%d.npy" % (mon, mon))
    print (feature_emb.shape)
    print (correlation_emb.shape)
#ele_dict = {}
    with open("./data/esrd_trans.pkl", "rb") as f:
        bird_dict = pickle.load(f)
#for i in range(cov.shape[0]):
#    ele_dict[i] = "a" + str(i)
    print ("1111")
    show_embed_correlation(correlation_emb / 1e5, bird_dict, mon)
    print ("2222")
    show_embed_feature(feature_emb / 1e5, bird_dict, mon)














