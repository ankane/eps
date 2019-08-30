import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import FunctionTransformer

df = pd.read_csv("test/support/data/mpg.csv")
# print(df)

X = df[['displ', 'year', 'cyl', 'trans', 'drv', 'class', 'manufacturer', 'model']]
y = df['hwy']

numeric_features = [0, 1, 2]
categorical_features = [3, 4, 5]
text_features = [6, 7]

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

text_transformer = Pipeline(steps=[
    ('vect', CountVectorizer())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ] + [('txt' + str(i), text_transformer, i) for i in text_features]
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
initial_type = [('numfeat', FloatTensorType([-1, 3])),
                ('catfeat', StringTensorType([-1, 3])),
                ('txtfeat', StringTensorType([-1, 2]))]
                # ('txtfeat2', StringTensorType([-1, 1]))]
onnx = convert_sklearn(pipe, initial_types=initial_type)
with open("test/support/linear_regression/mpg.onnx", "wb") as f:
  f.write(onnx.SerializeToString())

# from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer
# pydot_graph = GetPydotGraph(onnx.graph, name=onnx.graph.name, rankdir="TP",
#                             node_producer=GetOpNodeProducer("docstring"))
# pydot_graph.write_dot("graph.dot")

# import os
# os.system('dot -O -Tpng graph.dot')

import numpy
import onnxruntime as rt
sess = rt.InferenceSession("test/support/linear_regression/mpg.onnx")
print("input")
[print((o.name, o.type)) for o in sess.get_inputs()]
print("output")
[print((o.name, o.type)) for o in sess.get_outputs()]
input = {
    'numfeat': X[['displ', 'year', 'cyl']][:1].to_numpy(dtype=np.float32),
    'catfeat': X[['trans', 'drv', 'class']][:1].to_numpy(dtype=str),
    'txtfeat': X[['manufacturer', 'model']][:1].to_numpy(dtype=str)
}
res = sess.run(None, input)
print(res)
