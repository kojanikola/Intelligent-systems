import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

pd.set_option('display.width', 11)
pd.set_option('display.width', None)

data = pd.read_csv('datasets/house_prices_train.csv')

print(data.head())
print("-----------------------------------------------------------------")
print(data.tail())

print("*****************************************************************")

print(data.info())
print("-----------------------------------------------------------------")
print(data.describe())
print("-----------------------------------------------------------------")
# print(data.describe(include=[object]))
print("*****************************************************************")

X1 = data.loc[:, ['Year_built']]
X2 = data.loc[:, ['Area']]
X3 = data.loc[:, ['Bath_no']]
X4 = data.loc[:, ['Bedroom_no']]
y1 = data['Price']

y1 = y1 / 1000
print(y1)

plt.subplot(2, 3, 1)
plt.xlabel('Year_built', fontsize=12)
plt.scatter(X1, y1, edgecolors='black', marker='o')

plt.subplot(2, 3, 2)
plt.xlabel('Area', fontsize=12)
plt.scatter(X2, y1, edgecolors='black', marker='o')

plt.subplot(2, 3, 3)
plt.xlabel('Bath_no', fontsize=12)
plt.scatter(X3, y1, edgecolors='black', marker='o')

plt.subplot(2, 3, 4)
plt.xlabel('Bedroom_no', fontsize=12)
plt.scatter(X4, y1, edgecolors='black', marker='o')

plt.tight_layout()
plt.show()


# plt.figure()
# sb.heatmap(data.corr(), annot=True, fmt='.2f')
# plt.tight_layout()
# plt.show()

class LinearRegressionGradientDescent:
    def __init__(self):
        self.coeff = None
        self.features = None
        self.target = None
        self.mse_history = None

    def set_coefficients(self, *args):
        self.coeff = np.array(args).reshape(-1, 1)

    def cost(self):
        predicted = self.features.dot(self.coeff)
        s = pow(predicted - self.target, 2).sum()
        return (0.5 / len(self.features)) * s

    def predict(self, features):
        features = features.copy(deep=True)
        features.insert(0, 'c0', np.ones((len(features), 1)))
        features = features.to_numpy()
        print(features.dot(self.coeff))
        return features.dot(self.coeff).reshape(-1, 1).flatten()

    def gradient_descent_step(self, learning_rate):
        predicted = self.features.dot(self.coeff)
        s = self.features.T.dot(predicted - self.target)
        gradient = (1. / len(self.features)) * s
        self.coeff = self.coeff - learning_rate * gradient
        return self.coeff, self.cost()

    def perform_gradient_descent(self, learning_rate, num_iterations=100):
        self.mse_history = []
        for i in range(num_iterations):
            _, curr_cost = self.gradient_descent_step(learning_rate)
            self.mse_history.append(curr_cost)
        return self.coeff, self.mse_history

    def fit(self, features, target):
        self.features = features.copy(deep=True)

        coeff_shape = len(features.columns) + 1
        self.coeff = np.zeros(shape=coeff_shape).reshape(-1, 1)
        self.features.insert(0, 'c0', np.ones((len(features), 1)))
        self.features = self.features.to_numpy()
        self.target = target.to_numpy().reshape(-1, 1)


data_train = data.drop(columns=['Price'])

spots = 20
estates1 = pd.DataFrame(data=np.linspace(min(X1['Year_built']), max(X1['Year_built']), num=spots))
estates2 = pd.DataFrame(data=np.linspace(min(X2['Area']), max(X2['Area']), num=spots))
estates3 = pd.DataFrame(data=np.linspace(min(X3['Bath_no']), max(X3['Bath_no']), num=spots))
estates4 = pd.DataFrame(data=np.linspace(min(X4['Bedroom_no']), max(X4['Bedroom_no']), num=spots))
estates = pd.concat([estates1[0], estates2[0], estates3[0], estates4[0]], axis=1, keys=[0, 1, 2, 3])

lrgd = LinearRegressionGradientDescent()
lrgd.fit(data_train, y1)
learning_rates = np.array([[0.01], [0.01], [0.01], [0.01], [0.01]])

res_coeff, mse_history = lrgd.perform_gradient_descent(learning_rates, 20)

print(lrgd.predict(estates))

plt.figure(1)
line, = plt.plot(estates[0], lrgd.predict(estates), lw=5, c='red')
line.set_label('LRGD model')
plt.show()

data_train1 = data.drop(columns=['Price', 'Bath_no'])

print(data_train1)

lr_model = LinearRegression()
lr_model.fit(data_train, y1)

# Vizuelizacija modela
line, = plt.plot(estates[0], lr_model.predict(estates), lw=2, c='blue')
line.set_label('Ordinary LR model')

plt.legend(loc='upper left')
plt.show()
print(f'LRGD score: {lr_model.score(data_train, y1):.2f}')
