import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import math
from tensorflow import keras

#setting seed
np.random.seed(0)
tf.random.set_seed(0)

tfpl = tfp.layers
tfd = tfp.distributions


# Data Generation



def random_vector(N):
    m = np.random.uniform(0.0000001, 1.0, (N, 4))
    return m

def f1(x): return np.sin(x)
def f2(x): return x**(-0.5)
def f3(x): return 3*x + 1
def f4(x): return np.exp(3*x)
functions = [f1, f2, f3, f4]

def compute_y(xv):
    fx = np.array([functions[i](xv[:, i]) for i in range(4)]).T   # (N,4)
    fxs = fx @ np.ones((4, 1))                                    # somma colonne -> (N,1)
    noise = np.random.normal(0, 1, (fxs.shape[0], 1))             # rumore gaussiano
    y = fxs + noise
    return y

def average_squared_error(residuals):
    sqres = residuals ** 2
    n = sqres.shape[0]
    nor = np.full(n, 1/n)
    ase = sqres.T @ nor
    return ase

# Bayesian NN Model

# We choose a negative log likilihood as loss function 
# A BNN has to estime a distribution not a value so we use the nll as loss function
def negative_log_likelihood(y_true, y_pred_dist):
    return -y_pred_dist.log_prob(y_true)

# creating a function for BNN architecture
def build_bnn(input_shape, num_train_examples):

    # Define a custom KL divergence scaling function.
    # q = posterior distribution, p = prior distribution.
    def scaled_kl(q, p, _):
        # Return the KL divergence normalized by the number of training examples.
        # dividing by num_train_examples makes the KL term comparable
        # in scale to the data-fit term (likelihood)
        return tfd.kl_divergence(q, p) / num_train_examples

    inputs = keras.Input(shape=input_shape)
    # we use the KL to the loss function
    x = tfpl.DenseFlipout(
        32, activation='relu',
        kernel_divergence_fn=scaled_kl,
        bias_divergence_fn=scaled_kl
    )(inputs)

    x = tfpl.DenseFlipout(
        16, activation='relu',
        kernel_divergence_fn=scaled_kl,
        bias_divergence_fn=scaled_kl
    )(x)
    # compute the parameters dimension of a guassian (2)
    params_size = tfpl.IndependentNormal.params_size(event_shape=1)
    x = tfpl.DenseFlipout(
        params_size,
        kernel_divergence_fn=scaled_kl,
        bias_divergence_fn=scaled_kl
    )(x)
    # it takes the values from the hidden layer and trasform them in a Gaussina
    outputs = tfpl.IndependentNormal(event_shape=1)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# ##############
# Data Generating
# ##############

N_train = 10000
N_test  = 2000

V = random_vector(N_train)
Y = compute_y(V)

V_test = random_vector(N_test)
Y_test = compute_y(V_test)


# The data are scaled to improve the stability and robustness
#  of the training process, ensuring well-behaved gradients 
# and preventing numerical instabilities, especially in the Bayesian model

X_mean = V.mean(axis=0, keepdims=True)
X_std  = V.std(axis=0, keepdims=True)
V_scaled = (V - X_mean) / X_std
V_test_scaled = (V_test - X_mean) / X_std

Y_mean = Y.mean(axis=0, keepdims=True)   
Y_std  = Y.std(axis=0, keepdims=True)
Y_scaled = (Y - Y_mean) / Y_std
Y_test_scaled = (Y_test - Y_mean) / Y_std

# Plot with scaled Data

for i in range(4):
    plt.figure()
    plt.scatter(V[:, i], Y, s=5)
    plt.title(f"Column {i} vs Y")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

# Classic NN

model = keras.Sequential([
    keras.layers.Input(shape=(4,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1)
])

# we use adam optimizer and not gradient descend because it has better performance
# the algorithm work also with gd
model.compile(optimizer='adam', loss='mse')
model.fit(V_scaled, Y_scaled, epochs=50, batch_size=64, verbose=1)

Y_pred_scaled = model.predict(V_test_scaled)
# I descale the data
Y_pred = Y_pred_scaled * Y_std + Y_mean

# computing the residuals
residuals = Y_pred - Y_test

plt.figure()
plt.scatter(range(len(residuals)), residuals, s=5)
plt.axhline(0, color='red')
plt.title("Classic NN residuals distribution")
plt.grid(True)

ase_classic = average_squared_error(residuals)
print(f"Average Squared Error Classic NN: {ase_classic}")

# ######################################
# BNN
# #######################################

modelbay = build_bnn(input_shape=(4,), num_train_examples=N_train)
modelbay.compile(optimizer='adam', loss=negative_log_likelihood)

# Bayesian Neural Network need more training
history = modelbay.fit(
    V_scaled, Y_scaled,
    epochs=200,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

# Prediction BNN
# We use Monte Carlo Method

Nsample = 100
predictions_scaled = []

for _ in range(Nsample):
    y_dist = modelbay(V_test_scaled)            
    y_sample_scaled = y_dist.sample()          
    predictions_scaled.append(y_sample_scaled.numpy().flatten())

predictions_scaled = np.array(predictions_scaled) 

# scaled mean and variance 
mean_prediction_scaled = np.mean(predictions_scaled, axis=0).reshape(-1, 1)
std_prediction_scaled  = np.std(predictions_scaled, axis=0).reshape(-1, 1)

# descalde mean and variance
mean_prediction = mean_prediction_scaled * Y_std + Y_mean
std_prediction  = std_prediction_scaled * Y_std      

bayesian_residuals = mean_prediction - Y_test
ase_bnn = average_squared_error(bayesian_residuals)

print(f"Average Squared Error BNN : {ase_bnn}")
print(f"Average Squared Error NN classica : {ase_classic}")


# Evaluation BNN
# Bayesian NN has more tool to be evalueted, because it gives us a distribution and not a point estimation

# 1) Coverage 68% e 95%
# We compare the empirical confidence intervals with the theoretical ones.
# If the empirical are tighter means that the model is overconfindent, in the opposite
# way it is underconfident
z_68 = 1.0
z_95 = 1.96

lower_68 = mean_prediction - z_68 * std_prediction
upper_68 = mean_prediction + z_68 * std_prediction
lower_95 = mean_prediction - z_95 * std_prediction
upper_95 = mean_prediction + z_95 * std_prediction

inside_68 = (Y_test >= lower_68) & (Y_test <= upper_68)
inside_95 = (Y_test >= lower_95) & (Y_test <= upper_95)

coverage_68 = np.mean(inside_68)
coverage_95 = np.mean(inside_95)

print(f"Emipirical coverage  ~68%: {coverage_68:.3f}")
print(f"Empirical Coverage ~95%: {coverage_95:.3f}")

# 2) Sharpness: 
mean_std = np.mean(std_prediction)
print(f"The mean of the std. deviation is: {mean_std:.3f}")


# 3) Grafical visualisation with 95% confidence interval
# I create an array of length y.length 
idx = np.arange(len(Y_test))

plt.figure(figsize=(10,5))
plt.scatter(idx, Y_test, s=5, label="True Y")
plt.plot(idx, mean_prediction, label="Mean prediction BNN")
plt.fill_between(
    idx,
    lower_95.flatten(),
    upper_95.flatten(),
    alpha=0.3,
    label="Interval 95% BNN"
)
plt.xlabel("Idx sample test")
plt.ylabel("Y")
plt.title("BNN: prediction on a 95% confidence interval")
plt.grid(True)
plt.legend()

# 4) Graphic residual vs std.error
abs_residuals = np.abs(bayesian_residuals)

plt.figure(figsize=(6,5))
plt.scatter(std_prediction, abs_residuals, s=5)
plt.xlabel("std precicted by BNN")
plt.ylabel("Residual")
plt.grid(True)

plt.show()
