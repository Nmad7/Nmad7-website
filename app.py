from keras.models import load_model
import tensorflow as tf
from numpy import zeros
from numpy.random import randn
from matplotlib import pyplot as plt
import time
import os
from flask import Flask, render_template

def load():
    """Load in the pre-trained model"""
    global model
    model = load_model('generator_model_150.h5',compile=False)
    # Required for model to work
    global graph
    graph = tf.compat.v1.get_default_graph()
    return model

def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

def generate_data(g_model, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = g_model.predict(x_input)
    return X

def data_to_image(X):
    # scale from [-1,1] to [0,1]
    X = (X + 1) / 2.0
    # plot images
    for i in range(3 * 3):
        # define subplot
        plt.subplot(3, 3, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(X[i])
    # remove previous image
    for filename in os.listdir('static/'):
        if filename.startswith('cat'):  # not to remove other images
            os.remove('static/' + filename)
    # save new image
    new_image_name = "cat_" + str(time.time()) + ".jpg"
    plt.savefig('static/' + new_image_name)
    return new_image_name

def generate_image(gmodel):
    data = generate_data(gmodel, 100, 9)
    name = data_to_image(data)
    return name

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/cats")
def cats():
    gmodel = load()
    name = generate_image(gmodel)
    return render_template('cats.html')

@app.route("/getimage")
def get_img():
    name = generate_image(gmodel)
    return name


if __name__ == '__main__':
    app.run()