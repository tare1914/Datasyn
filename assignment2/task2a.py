import numpy as np
import utils
import typing
np.random.seed(1)


def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"
    # TODO implement this function (Task 2a)

    mean_sum = 0
    sta_dev_sum = 0
    for im in X: 
        mean_sum += np.mean(im)
        sta_dev_sum += np.std(im)

    mean = mean_sum/X.shape[0]
    sta_dev = sta_dev_sum/X.shape[0]

    X = (X - mean)/sta_dev
    
    bias = np.ones((X.shape[0],1))

    X = np.append(X, bias, axis=1)
    return X


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, 1]
        outputs: outputs of model of shape: [batch size, 1]
    Returns:
        Cross entropy error (float)
    """
    # TODO implement this function (Task 2a)
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"

    C = np.sum(-np.sum(targets*np.log(outputs),axis=1)) / targets.shape[0]

    return C


class SoftmaxModel:

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3a hyperparameter
                 use_improved_weight_init: bool  # Task 3c hyperparameter
                 ):
        # Always reset random seed before weight init to get comparable results.
        np.random.seed(1)
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # Initialize the weights
        self.ws = []
        prev = self.I

        self.hidden_layer_output = [] 
        self.hidden_layer_z = []

        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            if use_improved_weight_init:
                w = w = np.random.normal(0, 1/np.sqrt(prev), w_shape)
            else:
                w = np.random.uniform(-1, 1, w_shape)
            self.ws.append(w)
            prev = size
            if size != self.neurons_per_layer[len(self.neurons_per_layer)-1]:
                self.hidden_layer_output.append(np.zeros(size))
                self.hidden_layer_z.append(np.zeros(size))

        self.num_hidden_layers = len(self.hidden_layer_output)
        self.grads = [None for i in range(len(self.ws))]
        

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        # TODO implement this function (Task 2b)
        # HINT: For peforming the backward pass, you can save intermediate activations in varialbes in the forward pass.
        # such as self.hidden_layer_ouput = ...

        lay = 0 
        z = X.dot(self.ws[0])

        for weights in self.ws:
            if lay < self.num_hidden_layers:
                self.hidden_layer_z[lay] = z

                if self.use_improved_sigmoid:
                    self.hidden_layer_output[lay] = 1.7159*np.tanh((2.0/3.0)*z)
                else: 
                    self.hidden_layer_output[lay] = np.exp(z)/(np.exp(z)+1)

                z = self.hidden_layer_output[lay].dot(self.ws[lay + 1])
                lay += 1
            else:  
                e_z = np.exp(z)
                sum_zk = np.sum(e_z, axis=1, keepdims=True)
                
        return e_z / sum_zk
        

    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        # TODO implement this function (Task 2b)

        mean = 1 / X.shape[0]
        delta = outputs - targets
        
        self.grads[self.num_hidden_layers] = mean*np.transpose(self.hidden_layer_output[self.num_hidden_layers-1]).dot((delta)) 
         
        for i in reversed(range(self.num_hidden_layers)):
            if self.use_improved_sigmoid:
                sigmoid_derivate = (2*1.7159)/(3*np.power(np.cosh((2/3)*self.hidden_layer_z[i]),2))
            else:
                sigmoid_derivate = self.hidden_layer_output[i]*(1-self.hidden_layer_output[i])

            ws_err = self.ws[i+1].dot(delta.T)
            delta = sigmoid_derivate*(ws_err.T)

            if not i:
                self.grads[i] = mean*(X.T.dot(delta))
            else:
                self.grads[i] = mean*(self.hidden_layer_output[i-1].T.dot(delta))

        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"

        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer
        #self.grads = []

        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."

    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    # TODO: Implement this function (copy from last assignment)
    Ys = np.zeros((Y.shape[0], num_classes), dtype=int)
    
    for i in range(Y.shape[0]):
        Ys[i, Y[i]] = 1

    return Ys


def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited. 
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**2,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"


if __name__ == "__main__":
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = True
    use_improved_weight_init = True
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)
