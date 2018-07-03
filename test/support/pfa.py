from sklearn import linear_model
import titus.prettypfa
import json

x = [1, 2, 3, 5, 6]
y = [5 * xi + 3 for xi in x]

model = linear_model.LinearRegression()
model.fit([[xi] for xi in x], y)

def pfa(estimator):
    pfaDocument = titus.prettypfa.jsonNode('''
types:
  Regression = record(Regression,
                      const: double,
                      coeff: array(double))
input: array(double)
output: double
cells:
  regression(Regression) = {const: 0.0, coeff: []}
action:
  model.reg.linear(input, regression)
''')

    pfaDocument["cells"]["regression"]["init"] = {"const": estimator.intercept_, "coeff": list(estimator.coef_)}

    return pfaDocument

print(json.dumps(pfa(model)))
