import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.decoration import ContinuousDomain, CategoricalDomain
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml.feature_extraction.text import Splitter
from sklearn_pandas import DataFrameMapper

data = pd.read_csv("test/support/mpg.csv")

numeric_features = ["displ", "year", "cyl"]
categorical_features = ["class"]
text_features = []

mapper = DataFrameMapper(
  [(numeric_features, [ContinuousDomain()])] +
  [([f], [CategoricalDomain(), OneHotEncoder()]) for f in categorical_features] +
  [(f, [CategoricalDomain(), CountVectorizer(tokenizer=Splitter())]) for f in text_features]
)

pipeline = PMMLPipeline([
  ("mapper", mapper),
  ("model", GaussianNB())
])
pipeline.fit(data, data["drv"])

sklearn2pmml(pipeline, "test/support/python/naive_bayes.pmml")

print(pipeline.predict(data[:10]))
