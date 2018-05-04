import os
from functools import partial

import numpy as np
import tensorflow as tf
import edward as ed

from edward.models import Normal

def fully_connected(W, b, activation, X):
    Z = tf.matmul(X, W) + b
    return Z if activation is None else activation(Z)

def sequential(model):
    def sequential_model(X):
        for layer in model:
            X = layer(X)
        return tf.reshape(X, [-1])
    return sequential_model

def build_weights(layer_index, input_dim, output_dim):
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

def build_model(X_train, neurons):
    N, D = X_train.shape

    W_0, qW_0, b_0, qb_0 = build_weights(0, D, neurons)
    W_1, qW_1, b_1, qb_1 = build_weights(1, neurons, neurons)
    W_2, qW_2, b_2, qb_2 = build_weights(2, neurons, 1)

    model = sequential([
        partial(fully_connected, W_0, b_0, tf.tanh),
        partial(fully_connected, W_1, b_1, tf.tanh),
        partial(fully_connected, W_2, b_2, None),
    ])

    X = tf.cast(X_train, dtype=tf.float32)
    y = Normal(loc=model(X), scale=0.1 * tf.ones(N))

    parameters = {
        W_0: qW_0,
        b_0: qb_0,
        W_1: qW_1,
        b_1: qb_1,
    }

    return parameters, y

def build_inference(X_train, y_train, neurons=10):
    parameters, y = build_model(X_train, neurons)
    return ed.KLqp(parameters, data={y: y_train})

def run_inference(inference, num_iters=1000, logdir="./log"):
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    inference.run(n_iter=num_iters, logdir=logdir)

def generate_toy_data(N=40, noise_sd=0.1):
    raw_X = np.concatenate([
        np.linspace(0, 2, num=N/2),
        np.linspace(6, 8, num=N/2),
    ])
    y = np.cos(raw_X) + np.random.normal(0, noise_sd, size=N)
    X = (raw_X - 4.0)/4.0
    return X.reshape((N, 1)), y

def main(_):
    X_train, y_train = generate_toy_data()
    inference = build_inference(X_train, y_train)
    run_inference(inference)

if __name__ == "__main__":
    tf.app.run()
