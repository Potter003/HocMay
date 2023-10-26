import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing datasets
data_set = pd.read_csv('D:/HocMayVaUngDung/TH/LAB3_dataset/lab3/data.txt')
print(data_set.head())

#Extracting Independent and dependent variable
x = data_set.iloc[:, [2,3]].values
y = data_set.iloc[:, 3].values

#Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
print(x_train.shape)
print(y_train.shape)

#feature Scaling
from sklearn.preprocessing import StandardScaler
st_x = StandardScaler()
x_train = st_x.fit_transform(x_train)
x_test = st_x.transform(x_test)

# Fitting K-NN classifier to the training set
# n_neighbors: To define the required neighbors of the algorithm. Usually, it takes 5.
# metric = 'ninkowski': This is the default parameter and it decides the distance between the points.
# p=2: It is equivalent to the standard Euclidean metric.
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(x_train, y_train)

# Predicting the test set result
y_pred = classifier.predict(x_test)

# Creating the Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

#truc quan hoa K-NN tren bo train
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(np.arange(start= x_set[:, 1].min() - 1, stop= x_set[:, 0].max() + 1, step= 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop= x_set[:, 1].max() + 1, step= 0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN Algorithm (Training set)')
plt.xlabel('Age')
plt.ylabel('Class')
plt.legend()
plt.show()

#truc quan hoa K-NN tren bo test
from matplotlib.colors import ListedColormap
x_set, y_set = x_test, y_test
x1, x2 = np.meshgrid(np.arange(start= x_set[:, 1].min() - 1, stop= x_set[:, 0].max() + 1, step= 0.01),
                     np.arange(start= x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step=0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0],x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN algorithm(Test set)')
plt.xlabel('Age')
plt.ylabel('Class')
plt.legend()
plt.show()