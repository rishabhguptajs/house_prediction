import numpy as np
import matplotlib.pyplot as plt

plt.style.use('./deeplearning.mplstyle')

x_train = np.array([1.0, 1.2, 1.24, 1.56, 1.67, 1.89, 2.0])
y_train = np.array([300.0, 320.80, 330.0, 420.0, 434.90, 470.50, 500.0])

# print(f"x_train = {x_train}")
# print(f"y_train = {y_train}")

# print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]

# print(f"Number of training examples: {m}")

i = 0

x_i = x_train[i]
y_i = y_train[i]
# print(f"(x^({i}), (y^({i})) = ({x_i}, {y_i})")

# plt.show()

w = 200
b = 100

def compute_model_output(x, w, b):

    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w*x[i] + b

    return f_wb


tmp_f_wb = compute_model_output(x_train, w, b)

plt.plot(x_train, tmp_f_wb, c='b', label='Our Prediction')

plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')

plt.title("Housing Prices")
plt.ylabel("Prices (in 1000s of dollars)")
plt.xlabel("Size (1000 sqft)")

plt.legend()
plt.show()