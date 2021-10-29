import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# 1. PROBLEM STATEMENT AND DATA READING---------------------

pd.set_option('display.width', None)

data = pd.read_csv('datasets/cakes_train.csv')

print(data.head())
print("-----------------------------------------------------------------")
print(data.tail())

print("*****************************************************************")

# 2. DATA ANALYSIS

print(data.info())
print("-----------------------------------------------------------------")
print(data.describe())

print("*****************************************************************")

X1 = data.loc[:, ['flour']]
X2 = data.loc[:, ['eggs']]
X3 = data.loc[:, ['sugar']]
X4 = data.loc[:, ['milk']]
X5 = data.loc[:, ['butter']]
X6 = data.loc[:, ['baking_powder']]
y1 = data['type']

plt.subplot(2, 3, 1)
plt.xlabel('Grams of flour', fontsize=12)
plt.scatter(X1, y1, edgecolors='black', marker='o')

plt.subplot(2, 3, 2)
plt.xlabel('Eggs', fontsize=12)
plt.scatter(X2, y1, edgecolors='black', marker='o')

plt.subplot(2, 3, 3)
plt.xlabel('Grams of sugar', fontsize=12)
plt.scatter(X3, y1, edgecolors='black', marker='o')

plt.subplot(2, 3, 4)
plt.xlabel('Milk', fontsize=12)
plt.scatter(X4, y1, edgecolors='black', marker='o')

plt.subplot(2, 3, 5)
plt.xlabel('Butter', fontsize=12)
plt.scatter(X5, y1, edgecolors='black', marker='o')

plt.subplot(2, 3, 6)
plt.xlabel('Grams of baking_powder', fontsize=12)
plt.scatter(X6, y1, edgecolors='black', marker='o')

plt.tight_layout()
plt.show()

plt.figure()
sb.heatmap(data.corr(), annot=True, fmt='.2f')
plt.tight_layout()
plt.show()

# 3. DATA CLEANSING
# Nema nan vrednosti u kolonama sta tu treba ??????????????????????????

# 4. FEATURE ENGINEERING

data.eggs = data.eggs * 50

zbir = data.sum(axis=1)

data.flour = data.flour / zbir * 100
data.eggs = data.eggs / zbir * 100
data.sugar = data.sugar / zbir * 100
data.milk = data.milk / zbir * 100
data.butter = data.butter / zbir * 100
data.baking_powder = data.baking_powder / zbir * 100

print('6666666666666666')
print(data.to_string())

# data_train = data.loc[:, 'fluor', 'eggs', 'milk', 'butter', 'baking_powder']
# labels = data.loc[:, 'type']
# sve su brojne vrednosti je l treba?

# 5. MODEL TRAINING

dtc_model = DecisionTreeClassifier(criterion='entropy')

data_train = data.loc[:, ['flour', 'eggs', 'milk', 'butter', 'baking_powder']]
print("22222222222222")
print(data_train)
y = data['type']
X_train, X_test, y_train, y_test = train_test_split(data_train, y, train_size=0.7, random_state=123, shuffle=True)

dtc_model.fit(X_train, y_train)

labels_predicted = dtc_model.predict(X_test)

ser_pred = pd.Series(data=labels_predicted, name='Predicted', index=X_test.index)
res_df = pd.concat([X_test, y_test, ser_pred], axis=1)
print(res_df.head(10))

print(f'Model score: {dtc_model.score(X_test, y_test):0.3f}')

fig, axes = plt.subplots(1, 1, figsize=(8, 3), dpi=400)
tree.plot_tree(decision_tree=dtc_model, max_depth=10,
               feature_names=data_train.columns, class_names=['Muffin', 'Cupcake'],
               fontsize=3, filled=True)
fig.savefig('tree.png')
