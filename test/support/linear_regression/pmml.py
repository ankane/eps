from sklearn import linear_model
from scikit2pmml import scikit2pmml

x = [1, 2, 3, 5, 6]
y = [5 * xi + 3 for xi in x]

model = linear_model.LinearRegression()
model.fit([[xi] for xi in x], y)

scikit2pmml(estimator=model, file='pymodel.pmml')
