import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

df = pd.read_csv("test/support/data/houses.csv")
# print(df)

X = df[['bedrooms']]
y = df['price']

# categorical_features = ['state']
# categorical_transformer = Pipeline(steps=[
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
#     # ('ordinal', OrdinalEncoder())
# ])

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('cat', categorical_transformer, categorical_features)
#     ]
# )

estimators = [
  # ('preprocessor', preprocessor),
  ('regression', LinearRegression())
]
pipe = Pipeline(steps=estimators)
pipe.fit(X, y)
print(pipe.predict(X[:1]))

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
initial_type = [('x', FloatTensorType([-1, 1]))]
onx = convert_sklearn(pipe, initial_types=initial_type)
with open("test/support/linear_regression/houses.onnx", "wb") as f:
  f.write(onx.SerializeToString())

# import numpy
import onnxruntime as rt
sess = rt.InferenceSession("test/support/linear_regression/houses.onnx")
print("input")
[print((o.name, o.type)) for o in sess.get_inputs()]
print("output")
[print((o.name, o.type)) for o in sess.get_outputs()]
