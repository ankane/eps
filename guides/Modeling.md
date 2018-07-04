# Modeling

- [R JSON](#r-json)
- [R PMML](#r-pmml)
- [R PFA](#r-pfa)
- [Python JSON](#python-json)
- [Python PMML](#python-pmml)
- [Python PFA](#python-pfa)

## R JSON

Install the [jsonlite](https://cran.r-project.org/package=jsonlite) package

```r
install.packages("jsonlite")
```

And run:

```r
library(jsonlite)

model <- lm(dist ~ speed, cars)
data <- toJSON(list(coefficients=as.list(coef(model))), auto_unbox=TRUE)
write(data, file="model.json")
```

## R PMML

Install the [pmml](https://cran.r-project.org/package=pmml) package

```r
install.packages("pmml")
```

And run:

```r
library(pmml)

model <- lm(dist ~ speed,  cars)
data <- toString(pmml(model))
write(data, file="model.pmml")
```

## R PFA

Install the [aurelius](https://cran.r-project.org/package=aurelius) package

```r
install.packages("aurelius")
```

And run:

```r
library(aurelius)

model <- lm(dist ~ speed,  cars)
write_pfa(pfa(model), file="model.pfa")
```

## Python JSON

Run:

```python
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


data = json.dumps({'coefficients': coefficients})

with open('model.json', 'w') as f:
    f.write(data)
```

## Python PMML

Install the [scikit2pmml](https://github.com/vaclavcadek/scikit2pmml) package

```sh
pip install scikit2pmml
```

And run:

```python
from sklearn import linear_model
from scikit2pmml import scikit2pmml

x = [1, 2, 3, 5, 6]
y = [5 * xi + 3 for xi in x]

model = linear_model.LinearRegression()
model.fit([[xi] for xi in x], y)

scikit2pmml(estimator=model, file='model.pmml')
```

## Python PFA

Install the [Titus](https://github.com/opendatagroup/hadrian) package and run:

```python
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

data = json.dumps(pfa(model))

with open('model.pfa', 'w') as f:
    f.write(data)
```
