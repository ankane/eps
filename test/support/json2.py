from sklearn import linear_model
import pandas as pd
import json

x = [1, 2, 3, 4, 5]
y = [5 * xi + 3 for xi in x]

df = pd.DataFrame({'x': x, 'y': y})
features = ['x']

model = linear_model.LinearRegression()
model.fit(df[features], df['y'])

coefficients = {'_intercept': model.intercept_}
for i, c in enumerate(model.coef_):
    coefficients[features[i]] = c

print(json.dumps({'coefficients': coefficients}))
