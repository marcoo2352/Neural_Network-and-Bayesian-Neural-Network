import numpy as np 
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import random 
import math
from tensorflow import keras
#########################################################################
np.random.seed(0)
# Values Generetation
def random_vector(N):
    m = np.random.uniform(0,1,(N,4))
    return m

# Functions to apply to the values
def f1(x): return np.sin(x)
def f2(x): return x**(-0.5) 
def f3(x): return 3*x + 1
def f4(x): return np.exp(3*x)
functions = [f1, f2, f3, f4]

# Computation of y
def compute_y(xv):
    fx = np.array([functions[i](xv[:, i]) for i in range(4)]).T
    fxs = fx @ np.ones((4,1))
    noise = np.random.normal(0,1,(fxs.shape[0],1))
    y = fxs + noise
    return y

def average_squared_error(residuals):
    sqres = residuals ** 2
    nor = np.full(sqres.shape[0], 1/sqres.shape[0])
    ase = sqres.T @ nor
    return ase

tfpl = tfp.layers
tfd = tfp.distributions

def negative_log_likelihood(y_true, y_pred_dist):
    return -y_pred_dist.log_prob(y_true)


def build_bnn(input_shape):
    tfpl = tfp.layers
    tfd = tfp.distributions

    model_in = tf.keras.layers.Input(shape=input_shape)

    x = tfpl.DenseFlipout(32, activation='relu')(model_in)
    x = tfpl.DenseFlipout(16, activation='relu')(x)

    # layer che produce i parametri della Normal (mu e log_sigma)
    params_size = tfpl.IndependentNormal.params_size(event_shape=1)
    x = tfpl.DenseFlipout(params_size)(x)

    # layer che costruisce la distribuzione
    model_out = tfpl.IndependentNormal(event_shape=1)(x)

    model = tf.keras.Model(inputs=model_in, outputs=model_out)
    return model
#####################################################
V = random_vector(10000)
Y = compute_y(V)

for i in range(4):
    plt.figure()                
    plt.scatter(V[:, i], Y, s=5)
    plt.title(f"Colonna {i} vs Y")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)


##############################################
### Prediction with Classic Neural Network ###
##############################################

model = keras.Sequential([
    keras.layers.Input(shape=(4,)),     
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1)                   
])

model.compile(optimizer='adam', loss='mse')

model.fit(V, Y, epochs=1)  

V_test = random_vector(2000)
Y_test = compute_y(V_test)
Y_pred = model.predict(V_test)

# residuals evaluation

residuals = Y_pred - Y_test

plt.scatter(range(len(residuals)), residuals, s=5)
plt.axhline(0, color='red')
plt.xlabel("Indice campione")
plt.ylabel("Errore (residuo)")
plt.title("Residui del modello")
plt.grid(True)


# Average Squared Error
print(average_squared_error(residuals))


##############################################
##########   Bayesian Neural Networks ########
##############################################

modelbay = build_bnn(input_shape=(4,))
modelbay.compile(optimizer='adam', loss=negative_log_likelihood)
modelbay.fit(V, Y, epochs=300, batch_size=64, validation_split=0.2, verbose=1)
##
# Number of sample
Nsample = 100 
predictions = [] 

for _ in range(Nsample):
    y_dist = modelbay(V_test) 
    y_baypred_sample = y_dist.sample()
    predictions.append(y_baypred_sample.numpy().flatten())

predictions = np.array(predictions)

# Mean and std dev of prediction
mean_prediction = np.mean(predictions, axis=0).reshape(-1, 1)
std_prediction = np.std(predictions, axis=0).reshape(-1, 1)
#SS
bayesian_residuals =  mean_prediction - Y_test 
##############################################
#   VALUTAZIONE QUALITÀ DELLA BNN
##############################################

# 1) Calibrazione: copertura degli intervalli (68% e 95%)

z_68 = 1.0
z_95 = 1.96

lower_68 = mean_prediction - z_68 * std_prediction
upper_68 = mean_prediction + z_68 * std_prediction

lower_95 = mean_prediction - z_95 * std_prediction
upper_95 = mean_prediction + z_95 * std_prediction

# boolean mask di punti coperti dagli intervalli
inside_68 = (Y_test >= lower_68) & (Y_test <= upper_68)
inside_95 = (Y_test >= lower_95) & (Y_test <= upper_95)

coverage_68 = np.mean(inside_68)
coverage_95 = np.mean(inside_95)

print(f"Copertura empirica intervallo ~68%: {coverage_68:.3f}")
print(f"Copertura empirica intervallo ~95%: {coverage_95:.3f}")

# 2) Sharpness: quanto sono larghe le bande di incertezza

mean_std = np.mean(std_prediction)
print(f"Deviazione standard media predetta dalla BNN: {mean_std:.3f}")

# 3) Negative Log-Likelihood sul test (criterio naturale per BNN)

y_dist_test = modelbay(V_test)                # distribuzione predittiva sul test
log_prob_test = y_dist_test.log_prob(Y_test)  # log p(y | x)
nll_test = -tf.reduce_mean(log_prob_test)     # media della -log p(y|x)

print(f"Negative Log-Likelihood media sul test: {nll_test.numpy():.3f}")

# 4) Grafico: predizioni con intervallo al 95%

idx = np.arange(len(Y_test))

plt.figure(figsize=(10,5))
plt.scatter(idx, Y_test, s=5, label="Y vera")
plt.plot(idx, mean_prediction, label="Media predizione BNN")
plt.fill_between(
    idx,
    lower_95.flatten(),
    upper_95.flatten(),
    alpha=0.3,
    label="Intervallo 95% BNN"
)
plt.xlabel("Indice campione test")
plt.ylabel("Y")
plt.title("BNN: predizioni con intervallo di credibilità 95%")
plt.grid(True)
plt.legend()
plt.show()

# 5) Relazione tra incertezza e errore: |residuo| vs std

bayesian_residuals = mean_prediction - Y_test
abs_residuals = np.abs(bayesian_residuals)

plt.figure(figsize=(6,5))
plt.scatter(std_prediction, abs_residuals, s=5)
plt.xlabel("std predetta (incertezza BNN)")
plt.ylabel("|residuo| (errore assoluto)")
plt.title("Errore vs incertezza predetta")
plt.grid(True)
plt.show()