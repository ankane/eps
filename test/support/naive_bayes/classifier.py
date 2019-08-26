import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import GaussianNB
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
  ("classifier", GaussianNB())
])
pipeline.fit(df, df["color"])

sklearn2pmml(pipeline, "test/support/naive_bayes/houses.pmml")
