import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# ****    Generating Data   *****

np.random.seed(111)
x = np.linspace(-3, 3, 500)
x1 = np.linspace(0, 6, 500)
y = np.exp(np.sqrt(20-x**2) + np.random.normal(0, 0.2, 500))
y1 = -np.exp(np.sqrt(20-x**2) + np.random.normal(0, 0.2, 500)) + 80

data1 = pd.DataFrame(np.insert(x.reshape(-1, 1), 1, y, axis=1), columns=['X', 'Y'])
data1['Z'] = 0
data2 = pd.DataFrame(np.insert(x1.reshape(-1, 1), 1, y1, axis=1), columns=['X', 'Y'])
data2['Z'] = 1

data = pd.concat([data2, data1], axis=0)
#  **** Building The Model *******

y_data = pd.get_dummies(data['Z']).to_numpy()
x_train, x_test, y_train, y_test = train_test_split(data.iloc[:,:2].to_numpy(), y_data, train_size=0.8)

model = Sequential([
  layers.BatchNormalization(),
  layers.Dense(2, input_shape=x_train[0].shape),
  layers.Dense(16, activation='tanh'),
  layers.Dense(16, activation='tanh'),
  layers.Dense(2, activation='sigmoid')
])

model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=tf.keras.metrics.BinaryAccuracy())
model.fit(x_train, y_train, epochs=150, validation_data=(x_test, y_test))

# ****** Predicting Test Data *********

prediction = model.predict(x_test)
Z_predicted = []
for i in range(len(prediction)):
  Z_predicted.append(np.argmax(prediction[i]))

data2 = pd.DataFrame()
data2['X'] = x_test[:, 0]
data2['Y'] = x_test[:, 1]
data2['Z'] = y_test[:, 1]
data2['Z_predicted'] = Z_predicted


x3 = np.linspace(-3,6,1000)
y3 = np.random.random(1000)*220-60

prediction2 = model.predict(np.insert(x3.reshape(-1, 1), 1, y3, axis=1))
Z_predicted2 = []
for i in range(len(prediction2)):
  Z_predicted2.append(np.argmax(prediction2[i]))

data3 = pd.DataFrame()
data3['X'] = x3
data3['Y'] = y3
data3['Z'] = Z_predicted2



# ********* Visualisation ************


plt.figure(figsize=(10, 10))
plt.subplot(2,2,1)
sns.scatterplot(x='X', y='Y', data=data, hue='Z', markers='.')
plt.title('Real data')
plt.subplot(2,2,2)
sns.scatterplot(x='X', y='Y', data=data2, hue='Z', markers='.')
plt.title('Test data')
plt.subplot(2,2,3)
sns.scatterplot(x='X', y='Y', data=data2, hue='Z_predicted', markers='.')
plt.title('Test data classification')
plt.subplot(2,2,4)
sns.scatterplot(x='X', y='Y', data=data3, hue='Z', markers='.')
plt.title('Classifier')
plt.show()