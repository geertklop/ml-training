import numpy as np
from sklearn.metrics import accuracy_score
from .utils import sigmoid, load_planar_dataset


class NeuralNet:
    """Implementation of a n hidden layer network with SGD

    Args:
        dimensions: (tpl/ list) Dimensions of the neural net.
            (input, hidden layer, output)

    Attributes:
        n_layers: Number of hidden layers
        loss: Current loss
        loss_dict: Loss history (evcery 100th epoch)
        learning_rate: Learning rate parameter for updating weights
        w: Weights matrix per layer'
        b: Bias matrix per layer
        acitvations: activation function per layer
        z: linear outcome per layer (current)
        a: activated outcome per layer (current)
    """

    def __init__(self, dimensions):
        np.random.seed(2)

        self.n_layers = len(dimensions)
        self.loss = None
        self.loss_dict = {}
        self.learning_rate = None

        # initialize Weights & bias
        self.w = {}
        self.b = {}

        for i in range(1, self.n_layers):
            self.w[i] = np.random.randn(dimensions[i], dimensions[i - 1]) * 0.01
            self.b[i] = np.zeros((dimensions[i], 1))

        # Init activations & cache dicts
        self.activations = {1: np.tanh, 2: sigmoid}
        self.z = {}
        self.a = {}

    def _forward_propagation(self, X):
        """Single forward propagation pass"""
        self.z = {}
        self.a = {0: X}

        for i in range(1, self.n_layers):
            self.z[i] = np.dot(self.w[i], self.a[i - 1]) + self.b[i]
            self.a[i] = self.activations[i](self.z[i])

        return self.z, self.a

    def predict(self, X):
        """Feed forward pass and return output of final layer"""
        _, a = self._forward_propagation(X)

        return a[self.n_layers - 1]

    def _compute_cost(self, y_pred, y):
        """
        Computes the cross-entropy cost

        Args:
            y (array): Predicted outcomes
            y_pred (array): True labels

        Returns:
            cost (float): cross-entropy cost given equation
        """

        m = y.shape[1]  # number of training examples

        # Compute the cross-entropy cost
        logprobs = np.multiply(np.log(y_pred), y) + np.multiply(
            (1 - y), np.log(1 - y_pred)
        )

        cost = -np.sum(logprobs) / m

        # Make sure cost is right dimension. E.g., turns [[17]] into 17
        self.loss = np.squeeze(cost)

        assert isinstance(self.loss, float)

        return self.loss

    def _update_parameters(self, update_params):
        """Update weights and biases with gradient and learning rate"""
        for k, v in update_params.items():
            dw = v[0]
            db = v[1]

            self.w[k] -= self.learning_rate * dw
            self.b[k] -= self.learning_rate * db

    def _backward_propagation(self, X, y):
        """
        Args:
            X (array): input data of shape (2, number of examples)
            Y (array): labels vector of shape (1, number of examples)

        Returns:
        dict: gradients with respect to different parameters
        """
        m = X.shape[1]
        update_params = {}

        # start with dz for final layer
        dz = self.a[self.n_layers - 1] - y

        # loop over rest of layers
        for i in reversed(range(1, self.n_layers)):
            dw = (1 / m) * np.dot(dz, self.a[i - 1].T)
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

            if i > 1:
                dz = np.multiply(
                    np.dot(self.w[i].T, dz), 1 - np.power(self.a[i - 1], 2)
                )

            update_params[i] = (dw, db)

        return update_params

    def fit(self, X, y, epochs=10000, learning_rate=1.2, print_cost=True):
        """Fit model to input data.

        Implemented is a full training loop of:
        - Forward propagation
        - Cost calculation
        - Backpropagation
        - Update weights and biases using Stochastic Gradient Descent

        Args:
            X (array): Features vector
            y (array): Label vectors
            epochs (int, optional): Number of training rounds.
                Defaults to 10000.
            learning_rate (float, optional): Learning rate to use for
                gradient descent. Defaults to 1.2.
            print_cost (bool, optional): Whether to print the cost each
                1000th epoch. Defaults to True.
        """
        self.learning_rate = learning_rate
        # Loop (gradient descent)
        for i in range(0, epochs):
            # Forward propagation
            y_pred = self.predict(X)

            # Calculate loss
            cost = self._compute_cost(y_pred, y)

            # Backpropagation
            grads = self._backward_propagation(X, y)

            # Gradient descent parameter update.
            self._update_parameters(grads)

            # Print the cost every 1000 iterations
            if print_cost and i % 1000 == 0:
                self.loss_dict[i] = self.loss
                print("Cost after iteration %i: %f" % (i, cost))


if __name__ == "__main__":
    X_train, y_train = load_planar_dataset(m=1000)
    X_test, y_test = load_planar_dataset(m=400)

    n_x = X_train.shape[0]  # size of input layer
    n_h = 4
    n_y = y_train.shape[0]  # size of output layer

    nn = NeuralNet(dimensions=(n_x, n_h, n_y))
    nn.fit(X_train, y_train)

    preds = np.rint(nn.predict(X_train))
    print("Accuracy Train: {}".format(accuracy_score(y_train[0], preds[0])))

    preds = np.rint(nn.predict(X_test))
    print("Accuracy Test: {}".format(accuracy_score(y_test[0], preds[0])))
