import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.decoration import ContinuousDomain, CategoricalDomain
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn_pandas import DataFrameMapper

df = pd.read_csv("test/support/data/houses.csv")

pipeline = PMMLPipeline([
  ("mapper", DataFrameMapper([
    (["bedrooms", "bathrooms"], [ContinuousDomain()]),
    (["state"], [CategoricalDomain(), OneHotEncoder()])
  ])),
  ("regression", LinearRegression())
])
pipeline.fit(df, df["price"])

sklearn2pmml(pipeline, "test/support/linear_regression/houses.pmml")
