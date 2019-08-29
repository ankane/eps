import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

df = pd.read_csv("test/support/data/houses.csv")
# print(df)

X = df[['bedrooms', 'bathrooms', 'state', 'color']]
y = df['price']

numeric_features = [0, 1]
categorical_features = [2, 3]

# numeric_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='median'))
#     # ('scaler', StandardScaler())
# ])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    # ('ordinal', OrdinalEncoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

estimators = [
  ('preprocessor', preprocessor),
  ('regression', LinearRegression())
]
pipe = Pipeline(steps=estimators)
pipe.fit(X, y)
print(X[:1])
print(pipe.predict(X[:1]))

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
initial_type = [('numfeat', FloatTensorType([1, 2])),
                ('strfeat', StringTensorType([1, 2]))]
onnx = convert_sklearn(pipe, initial_types=initial_type)
with open("test/support/linear_regression/houses.onnx", "wb") as f:
  f.write(onnx.SerializeToString())


# from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer
# pydot_graph = GetPydotGraph(onnx.graph, name=onnx.graph.name, rankdir="TP",
#                             node_producer=GetOpNodeProducer("docstring"))
# pydot_graph.write_dot("graph.dot")

# import os
# os.system('dot -O -Tpng graph.dot')

# import numpy
import onnxruntime as rt
sess = rt.InferenceSession("test/support/linear_regression/houses.onnx")
print("input")
[print((o.name, o.type)) for o in sess.get_inputs()]
print("output")
[print((o.name, o.type)) for o in sess.get_outputs()]
print(X[['bedrooms', 'bathrooms']][:1].to_numpy())
print(X[['state', 'color']][:1].to_numpy(dtype=str))
res = sess.run(None, {'numfeat': X[['bedrooms', 'bathrooms']][:1].to_numpy(dtype=np.float32), 'strfeat': X[['state', 'color']][:1].to_numpy(dtype=str)})
print(res)
