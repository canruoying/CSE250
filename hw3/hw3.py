import numpy as np


def generatePoint(sigma):
    """Returns an example (X, y) with given sigma"""
    mu = 0.2
    sign = np.random.choice([-1, 1])
    MU = np.array([mu, mu, mu, mu, mu]) * sign
    X = sigma * np.random.randn(1, 5)[0] + MU
    y = sign
    return (X, y)

def generateData(sigma, project):
    """Returns a example (X, y) projected onto a domain set using the project function"""
    (X, y) = generatePoint(sigma)
    X = project(X)
    X = np.append(X, 1)
    return (X, y)

def projectCube(P):
    """Returns the vector P projected onto the hypercube"""
    Q = P
    for i in range(Q.size):
        if Q[i] >= 0.:
            Q[i] = min(1., Q[i])
        else:
            Q[i] = max(-1., Q[i])
    return Q

def projectBall(P):
    """Returns the vector P projected onto the unit ball"""
    Q = P
    norm = np.linalg.norm(Q)
    if norm > 1.:
        for i in range(Q.size):
            Q[i] = Q[i] / norm
    return Q

def logistic(x):
    """Returns the logistic function applied to the scalar input x"""
    return 1. / (1. + np.exp(-x)) - 0.5

def binaryError(pred, y):
    """Returns the binary classifier error evaluated using Indicator(Sign(pred,y))"""
    if (pred * y >= 0):
        return 0
    return 1

def getTestData(N, sigma, project):
    """Returns a test set of length N generated using given sigma and project function"""
    T_X = np.ones((N,6))
    T_y = np.ones(N)
    for i in range(N):
        (T_X[i], T_y[i]) = generateData(sigma, project)
    return (T_X, T_y)

def getAlpha(T, project):
    """Returns learning rate alpha calculated using M/(rho*sqrt(T))"""
    M_cube = 2 * np.sqrt(6)
    r_cube = np.sqrt(6)
    M_ball = 2
    r_ball = 1
    if project == projectCube:
        return M_cube / (r_cube * np.sqrt(T))
    if project == projectBall:
        return M_ball / (r_ball * np.sqrt(T))

def runExperiment(T, sigma, project):
    """Returns a weight W_hat trained with T examples and projected using project function"""
    alpha = getAlpha(T, project)
    W = np.array([0., 0., 0., 0., 0., 0.])
    W_hat = W / T
    for t in range(1, T):
        (X, y) = generateData(sigma, project)
        preds = logistic(X.dot(W))
        error = preds - y
        gradient = X * error
        W -= gradient * alpha
        project(W)
        W_hat += W / T
    return W_hat

def evaluate(W, X, y):
    """Returns the loss and error of weight W evaluated on sample set X and y"""
    N = X.shape[0]
    preds = X.dot(W)
    loss = 0
    error = 0
    for i in range(N):
        loss += np.log(1 + np.exp(- preds[i] * y[i])) / N
        error += binaryError(preds[i], y[i]) / N
    return (loss, error)

# Prepare output files
results = 'results.csv'
summary = 'summary.csv'
r = open(results, 'w')
s = open(summary, 'w')
r.write("scenario, sigma, T, run, loss, error\n")
s.write("scenario, sigma, T, loss_mean, loss_std, loss_min, excess, error_mean, error_stdev\n")

N = 400
runs = 30
sigmas = [0.05, 0.3]
Ts = [50, 100, 500, 1000]

# Run experiments on two scenarios
for project in [projectCube, projectBall]:
    # Run experiments on sigma = 0.05, 0.3
    for sigma in sigmas:
        # Generate a single test set for all experiments
        (TC_X, TC_y) = getTestData(N, sigma, project)
        # Run experiments on T = 50, 100, 500, 1000
        for T in Ts:
            # Initialize L (loss) and E (binary error) for 30 runs
            L = np.ones(runs)
            E = np.ones(runs)
            # Do 30 runs for each setting
            for i in range(runs):
                W = runExperiment(T, sigma, project)
                (L[i], E[i]) = evaluate(W, TC_X, TC_y)
                if project == projectCube:
                    scenario = "Cube"
                else:
                    scenario = "Ball"
                line = [str(scenario), ',', str(sigma), ',', str(T), ',', str(i), ',', str(L[i]), ',', str(E[i])]
                r.write(''.join(line))
                r.write('\n')
            line = [str(scenario), ',', str(sigma), ',', str(T), ',', str(L.mean()), ',', str(L.std()), ',', str(L.min()), ',', str(L.mean()-L.min()), ',', str(E.mean()), ',', str(E.std())]
            s.write(''.join(line))
            s.write('\n')
r.close()
s.close()
