from scipy.optimize import minimize
import matplotlib.pyplot as plt
from itertools import product
import numpy as np


def linear_kernel(x, y):
    """Linear kernel
    Args:
        x {np.array}
        y {np.array}
    Returns:
        {np.array}: Dot product of x and y
    """
    return np.dot(np.transpose(x), y)

def polynomial_kernel(x, y, p=2):
    """Polynomial kernel
    Args:
        x {np.array}
        y {np.array}
        p {float}: Order of the polynomial
    Returns:
        {np.array}: Polynomial kernel applied to x and y
    """
    return np.power(np.add(np.dot(x, y), 1), p)

def rbf_kernel(x, y, sigma=0.1):
    """Radial Basis Function kernel
    Args:
        x {np.array}
        y {np.array}
        sigma {float}: RBF width
    Returns:
        {np.array}: Radial Basis Function kernel applied to x and y
    """
    return np.exp(
        -np.power(np.linalg.norm(np.subtract(x, y)), 2.0) / (2.0 * np.power(sigma, 2.0))
    )


class SVM(object):

    def __init__(self, x, t, kernel="linear", kernel_param=None, slack_C=None):
        """SVM constructor
        Args:
            x {np.array}: Inputs
            t {np.array}: Targets
            kernel {string}: Type of kernel (default: "linear")
            kernel_param {float}: Parameter of the kernel function (default: None)
            slack_C {float}: Slack variables factor (default: None)
        """
        kernels = {
            "linear": linear_kernel,
            "polynomial": lambda x, t: polynomial_kernel(x, t, kernel_param),
            "rbf": lambda x, t: rbf_kernel(x, t, kernel_param)
        }
        self.x = x # Inputs
        self.t = t # Targets
        self.kernel = kernels[kernel] # Kernel function
        self.n = t.shape[-1] # Number of inputs
        self.p = self.compute_p() # Matrix computed once and used in every self.objective
        self.bounds = [(0, slack_C) for _ in range(self.n)] # Bounds for scipy.optimize.minimize
        self.alpha = None # Alphas to optimize (dual problem)
        self.sv_indexes = None # Support vectors indexes
        self.b = None # Threshold
    
    def compute_p(self):
        """Compute the matrix used in self.objective
        Returns:
            {np.array}: Matrix p to be used in self.objective
        """
        kernel = np.reshape(
            [self.kernel(x, y) for x, y in product(self.x, self.x)],
            (self.n, self.n)
        )
        return np.multiply(np.outer(self.t, self.t), kernel)

    def objective(self, alpha):
        """Objective function to minimize
        Args:
            alpha {np.array}: Dual problem factors
        Returns:
            {float}: Objective function result
        """
        return 0.5 * np.sum(np.multiply(np.outer(alpha, alpha), self.p)) - np.sum(alpha)

    def zerofun(self, alpha):
        """Constraint function for the minimization problem
        Args:
            alpha {np.array}: Dual problem factors
        Returns:
            {np.array}: Dot product of alpha and targets
        """
        return np.dot(alpha, self.t)
    
    def compute_b(self):
        """Compute the thresold b
        Returns:
            {float} SVM threshold b
        """
        alphas = self.alpha[self.sv_indexes]
        x = self.x[self.sv_indexes]
        t = self.t[self.sv_indexes]
        sv = self.x[self.sv_indexes][0]
        sv_t = self.t[self.sv_indexes][0]
        kernels = np.array([self.kernel(sv, xx) for xx in x])
        return np.sum(np.multiply(np.multiply(alphas, t), kernels)) - sv_t
    
    def fit(self, verbose=False):
        """Find the support vectors and compute the threshold
        Args:
            verbose {boolean}: true for printing minimize message, false otherwise (default: False)
        """
        xc = {
            "type": "eq",
            "fun": lambda alpha: self.zerofun(alpha)
        }
        obj = lambda alpha: self.objective(alpha)
        ret = minimize(obj, np.zeros(self.n), bounds=self.bounds, constraints=xc)
        self.alpha = ret["x"]
        self.sv_indexes = np.where(self.alpha > 1e-5)
        self.b = self.compute_b()
        if verbose:
            print(ret["message"])
        return ret["success"]
    
    def ind(self, new_data):
        """Indicator function
        Args:
            new_data {np.array}: data point we want to classify
        Returns:
            {float} Indicator value, deciding the class of new_data
        """
        alphas = self.alpha[self.sv_indexes]
        x = self.x[self.sv_indexes]
        t = self.t[self.sv_indexes]
        kernels = np.array([self.kernel(new_data, xx) for xx in x])
        return np.sum(np.multiply(np.multiply(alphas, t), kernels)) - self.b

    def plot_decision_boundary(self):
        """Plot the SVM decision boundary and its margin
        """
        xgrid = np.linspace(-5, 5)
        ygrid = np.linspace(-4, 4)
        grid = np.array(
            [[self.ind(np.array([x, y])) for x in xgrid] for y in ygrid]
        )
        plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), 
            colors=("red", "black", "blue"), linewidths=(1, 3, 1))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Decision boundary")


def gen_data(sizeA=0.2, sizeB=0.2, meanA1=[1.5, 0.5], meanA2=[-1.5, 0.5], meanB=[0.0, -0.5]):
    """Generate two classes of data
    Args:
        sizeA {float}: std of class A points
        sizeB {float}: std of class B points
        meanA1 {np.array}: 2D point, mean of first class A cluster
        meanA2 {np.array}: 2D point, mean of second class A cluster
        meanB {np.array}: 2D point, mean class B cluster
    Returns:
        classA {np.array}: set of points of the first class
        classB {np.array}: set of points of the second class
        inputs {np.array}: set of all points randomly shuffled 
        targets {np.array}: targets values matching inputs
    """
    classA = np.concatenate(
        (
            np.random.randn(10, 2) * sizeA + meanA1,
            np.random.randn(10, 2) * sizeA + meanA2
        )
    )
    classB = np.random.randn(20, 2) * sizeB + meanB
    inputs = np.concatenate((classA, classB))
    targets = np.concatenate(
        (
            np.ones(classA.shape[0]),
            -np.ones(classB.shape[0])
        )
    )
    N = inputs.shape[0] # Number of samples
    permute = np.arange(N)
    np.random.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]
    return classA, classB, inputs, targets


def plot_data(classA, classB, save=False):
    """Plot class a and classB
    Args:
        save {boolean}: true for saving fig into a pdf file, false otherwise (default: False)
    """
    plt.plot(
        [p[0] for p in classA],
        [p[1] for p in classA],
        "b.",
        label="class A"
    )
    plt.plot(
        [p[0] for p in classB],
        [p[1] for p in classB],
        "r.",
        label="class B"
    )
    plt.axis("equal")
    if save:
        plt.savefig("data.pdf")
    plt.legend(loc="upper right")


def main():
    classA, classB, inputs, targets = gen_data()
    svm = SVM(inputs, targets, kernel="rbf", kernel_param=1)
    if svm.fit(verbose=True):
        plot_data(classA, classB)
        svm.plot_decision_boundary()


if __name__ == "__main__":
    main()