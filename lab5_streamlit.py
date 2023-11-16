import streamlit as st
import pandas as pd
import numpy as np

st.write('Nguyễn Lê Thanh Duy')
st.write('2174802010358')
st.title('lab 5:')
st.code("""#Mo hinh Linear Regrsstion
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import matplotlib.axes as ax

data = pd.read_csv('D:\HocMayVaUngDung\TH/data_for_lr.csv')

#drop the missing values
data = data.dropna()

#training dataset and labels
train_input = np.array(data.x[0:500]).reshape(500,1)
train_output = np.array(data.y[0:500]).reshape(500,1)

#valid dataset and labels
test_input = np.array(data.x[500:700]).reshape(199,1)
test_output = np.array(data.y[500:700]).reshape(199,1)


class LinearRegression:
    def __init__(self):
        self.parameters = {}
    
    def forward_propagation(self, train_input):
        m = self.parameters['m']
        c = self.parameters['c']
        predictions = np.multiply(m, train_input) + c
        return predictions
    
    def cost_function(self, predictions, train_output):
        cost = np.mean((train_output - predictions) ** 2)
        return cost
    
    def backward_propagation(self, train_input, train_output, predictions):
        derivatives = {}
        df = (train_output - predictions) * -1
        dm = np.mean(np.multiply(train_output, df))
        dc = np.mean(df)
        derivatives['dm'] = dm
        derivatives['dc'] = dc
        return derivatives
    
    def update_parameters(self, derivatives, learning_rate):
        self.parameters['m'] = self.parameters['m'] - learning_rate * derivatives['dm']
        self.parameters['c'] = self.parameters['c'] - learning_rate * derivatives['dc']

    def train(self, train_input, train_output, learning_rate, iters):
        #initialize random parameters
        self.parameters['m'] = np.random.uniform(0, 1) * -1
        self.parameters['c'] = np.random.uniform(0, 1) * -1

        #initialize loss
        self.loss = []

        #iterate
        for i in range(iters):
            #forward propagation
            predictions = self.forward_propagation(train_input)

            #cost function
            cost = self.cost_function(predictions, train_output)

            #append loss and print
            self.loss.append(cost)
            print('Iteration = {}, Loss'.format(i + 1, cost))

            #back propagation
            derivatives = self.backward_propagation(train_input, train_output, predictions)

            #update parameters
            self.update_parameters(derivatives, learning_rate)
        return self.parameters, self.loss

linear_reg = LinearRegression()
parameters, loss = linear_reg.train(train_input, train_output, 0.0001, 20)

#Prediction on test data
y_pred = test_input * parameters['m'] + parameters['c']

#Plot the regesstion line with actual data pointa
plt.plot(test_input, test_output, '+' , label = 'Actual values')
plt.plot(test_input, y_pred, label = 'Predicted values')
plt.xlabel('Test input')
plt.ylabel('Test Output or Predicted output')
plt.legend()
plt.show()

#Mo hinh Polynomnal Regression
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

#Importing the dataset
datas = pd.read_csv('D:\HocMayVaUngDung\TH/data.csv')
print(datas.head())

#Features and the target variables
X = datas.iloc[:, 1:2].values
y = datas.iloc[:, 2].values

poly = PolynomialFeatures(degree= 4)
X_poly = poly.fit_transform(X)

poly.fit(X_poly, y)
lin2 = LinearRegression()
lin2.fit(X_poly, y)

#Visualising the Polynomial Regression results
plt.scatter(X, y, color= 'blue')

plt.plot(X, lin2.predict(poly.fit_transform(X)), color='red')
plt.title('Polymial Regression')
plt.xlabel('Temperature')
plt.ylabel('Pressure')
plt.show()
#Predicting a new result with Polynomial Regression
#after converting predict variable to 2D array
pred2 = 110.0
pred2array = np.array([[pred2]])
lin2.predict(poly.fit_transform(pred2array))
        """)