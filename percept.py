import numpy as np
import pandas as pd
import pdb
# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)
import os


def stepFunction(t):
    if t >= 0:
        return 1
    return 0


def prediction(X, W, b):
    return stepFunction((np.matmul(X, W)+b)[0])

# TODO: Fill in the code below to implement the perceptron trick.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.

def determine_increment_of_coefficient_and_weights(column):

  if column['y_hat'] > 0 and column['y'] > 0 or column['y_hat'] < 0 and column['y'] < 0:
    return 0

  if column['y_hat'] > 0 and column['y'] < 0:
    return -1
  
  return 1


def add_weights_to_x_axis(coefficient, boolean, learn_rate):

  pdb.set_trace()
 
  if boolean == 0:
    return 0
  
  if boolean == 1:
    return coefficient * learn_rate
  
  return coefficient * learn_rate * -1
 
  print(locals())

  
def increment_learn_rate(column, learn_rate):

  if column['coefficient_bool'] == 0:
    return 0

  if column['coefficient_bool'] == 1:
    return learn_rate
  
  return learn_rate * -1


def perceptronStep(X, y, W, b, learn_rate=0.01):

    X['X_0_W_0'] = X['x_coeff_1'] * W[0]
    X['X_1_W_1'] = X['x_coeff_2'] * W[1]

    y_hat = X['X_0_W_0'] + X['X_1_W_1'] + b

    values = pd.DataFrame({'y_hat': y_hat, 'y': y})

    X['coefficient_bool'] =  values.apply(lambda column: determine_increment_of_coefficient_and_weights(column), axis=1)
    X['learn_rate_increment'] = X.apply(lambda column: increment_learn_rate(column, learn_rate), axis=1)
    # finish creating weight adjustments, add weights to axis and return
    X['weight_adust_0'] = X.apply(lambda column: add_weights_to_x_axis(X['x_coeff_1'].values, X['coefficient_bool'].values, learn_rate), axis=1)
    X['weight_adust_1'] = X.apply(lambda column: add_weights_to_x_axis(X['x_coeff_2'], X['coefficient_bool'], learn_rate), axis=1)

    b += X['learn_rate_increment'].sum()
    print(W[0])
    return W, b

# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.


def trainPerceptronAlgorithm(X, y, learn_rate=0.01, num_epochs=1):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2, 1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines

def main():

  labelled_data = pd.read_csv(os.getcwd() + '/data.csv', delimiter=',', index_col=False, names=['x_coeff_1', 'x_coeff_2', 'y'])
  X = labelled_data[labelled_data.columns[0:-1]]
  y = labelled_data[labelled_data.columns[-1]]
  trainPerceptronAlgorithm(X, y)



main()