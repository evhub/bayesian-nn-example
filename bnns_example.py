"""
A toy Bayesian neural network example by Evan Hubinger.
Code borrowed heavily from <http://edwardlib.org/tutorials/>.
"""

import os
from functools import partial
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import edward as ed
from edward.models import Normal


def fully_connected(W, b, activation, X):
    """Tensorflow operation which acts as a fully-connected layer in a neural network."""
    Z = tf.matmul(X, W) + b
    return Z if activation is None else activation(Z)


def sequential_model(X, model):
    """Combine a list of partially-applied layers into a single model."""
    for layer in model:
        X = layer(X)
    return tf.reshape(X, [-1])


def build_weights(layer_index, input_dim, output_dim):
    """Return (weight model, weight param, bias model, bias param)."""
    W = Normal(
        loc=tf.zeros([input_dim, output_dim]),
        scale=tf.ones([input_dim, output_dim]),
    )
    qW = Normal(
        loc=tf.get_variable("qW_{}/loc".format(layer_index), [input_dim, output_dim]),
        scale=tf.nn.softplus(tf.get_variable("qW_{}/scale".format(layer_index), [input_dim, output_dim])),
    )

    b = Normal(
        loc=tf.zeros(output_dim),
        scale=tf.ones(output_dim),
    )
    qb = Normal(
        loc=tf.get_variable("qb_{}/loc".format(layer_index), [output_dim]),
        scale=tf.nn.softplus(tf.get_variable("qb_{}/scale".format(layer_index), [output_dim])),
    )

    return W, qW, b, qb


def build_nn(X, W_0, b_0, W_1, b_1, W_2, b_2):
    """Construct a neural network from the given parameters."""
    return sequential_model(X, [
        partial(fully_connected, W_0, b_0, tf.tanh),
        partial(fully_connected, W_1, b_1, tf.tanh),
        partial(fully_connected, W_2, b_2, None),
    ])


def estimate_noise(y_train, verbose=True):
    """Estimate the noise in the training data with a finite difference method."""
    N, = y_train.shape
    sig = np.std(np.diff(y_train))/2
    if verbose:
        print("sig ~=", sig)
    return sig


def build_model(X_train, sig, neurons=10, verbose=True):
    """Return (y model, parameters)."""
    N, D = X_train.shape

    W_0, qW_0, b_0, qb_0 = build_weights(0, D, neurons)
    W_1, qW_1, b_1, qb_1 = build_weights(1, neurons, neurons)
    W_2, qW_2, b_2, qb_2 = build_weights(2, neurons, 1)

    parameters = OrderedDict([
        (W_0, qW_0),
        (b_0, qb_0),
        (W_1, qW_1),
        (b_1, qb_1),
        (W_2, qW_2),
        (b_2, qb_2),
    ])

    X = tf.cast(X_train, dtype=tf.float32)
    y = Normal(
        loc=build_nn(X, *parameters.keys()),
        scale=sig * tf.ones(N),
    )

    return y, parameters


def build_inference(y, parameters, y_train):
    """Return inference engine."""
    return ed.KLqp(parameters, data={y: y_train})


def run_inference(inference, num_iters=1500, batch_size=15, logdir="./log"):
    """Run inference engine."""
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    inference.run(n_iter=num_iters, n_samples=batch_size, logdir=logdir)


def build_posterior(y, parameters):
    """Return y posterior."""
    return ed.copy(y, parameters)


def plot_samples(y_post, y_test, num_bins=20):
    """Plot y samples from the posterior alongside the actual y values."""
    def T(data, latent_params):
        return tf.reduce_mean(tf.cast(data[y_post], tf.float32))
    Ty_rep, Ty = ed.ppc(T, data={y_post: y_test})
    ed.ppc_stat_hist_plot(Ty, Ty_rep, bins=num_bins)
    plt.show()


def raw_predictions(y_true, y_pred):
    """Evaluation metric that just returns the raw predictions."""
    return y_pred


def plot_predictions(y_post, X_test, y_test):
    """Plot y together with y hat."""
    y_pred = ed.evaluate(raw_predictions, data={y_post: y_test})
    plt.scatter(X_test, y_test, label="truth")
    plt.scatter(X_test, y_pred, label="predicted")
    plt.legend()
    plt.show()


def build_sampler(parameters, X_test, num_samples=10, num_points=400):
    """Return (test inputs, model samplers)."""
    min_x = np.min(X_test)
    max_x = np.max(X_test)

    inputs = np.linspace(min_x, max_x, num=num_points, dtype=np.float32)
    X = tf.expand_dims(inputs, 1)

    samples = tf.stack([
        build_nn(X, *(param.sample() for param in parameters.values()))
        for _ in range(num_samples)
    ])

    return inputs, samples


SESSION = None


def init_session():
    """Initialize the Tensorflow session."""
    global SESSION
    if SESSION is None:
        SESSION = ed.get_session()
        tf.global_variables_initializer().run()


def eval_samples(samples):
    """Make draws from the given samplers."""
    init_session()
    return samples.eval()


def plot_model(inputs, samples, X_test, y_test):
    """Plot y together with various draws from the model."""
    plt.scatter(X_test, y_test, label="truth")
    for i, output in enumerate(eval_samples(samples)):
        plt.plot(inputs, output.T, label="model {}".format(i))
    plt.legend()
    plt.show()


def generate_toy_data(N=200, noise_sd=0.05, plot=False):
    """Generate (X, y)."""
    X = np.concatenate([
        np.linspace(-1, -0.1, num=N//2),
        np.linspace(0.1, 1, num=(N+1)//2),
    ])
    y = np.sin((X + 1)*np.pi) + np.random.normal(0, noise_sd, size=N)
    if plot:
        plt.scatter(X, y)
        plt.show()
    return X.reshape((N, 1)), y


def main(argv):
    """Generate toy data, run inference, and evaluate model."""
    # generate data
    plt.title("Training Data")
    X_train, y_train = generate_toy_data(plot=True)

    # construct model
    sig = estimate_noise(y_train)
    y, parameters = build_model(X_train, sig)

    # evaluate prior
    inputs, samples = build_sampler(parameters, X_train)

    plt.title("Prior Draws")
    plot_model(inputs, samples, X_train, y_train)

    # run inference
    inference = build_inference(y, parameters, y_train)
    run_inference(inference)

    # evaluate posterior
    plt.title("Posterior Draws")
    plot_model(inputs, samples, X_train, y_train)

    y_post = build_posterior(y, parameters)

    plt.title("Posterior Predictions")
    plot_predictions(y_post, X_train, y_train)

    plot_samples(y_post, y_train)


if __name__ == "__main__":
    tf.app.run(main)
