import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from lightgbm import LGBMClassifier
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.decoration import ContinuousDomain, CategoricalDomain
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml.preprocessing import PMMLLabelEncoder
from sklearn2pmml.feature_extraction.text import Splitter
from sklearn_pandas import DataFrameMapper

binary = False

data = pd.read_csv("test/support/mpg.csv")
if binary:
  data["drv"] = data["drv"].replace("r", "4")

numeric_features = ["displ", "year", "cyl"]
categorical_features = ["class"]
text_features = []

mapper = DataFrameMapper(
  [(numeric_features, [ContinuousDomain()])] +
  [([f], [CategoricalDomain(), PMMLLabelEncoder()]) for f in categorical_features] +
  [(f, [CategoricalDomain(), CountVectorizer(tokenizer=Splitter())]) for f in text_features]
)

pipeline = PMMLPipeline([
  ("mapper", mapper),
  ("model", LGBMClassifier(n_estimators=1000))
])
pipeline.fit(data, data["drv"], model__categorical_feature=[3])

suffix = "binary" if binary else "multiclass"
sklearn2pmml(pipeline, "test/support/python/lightgbm_" + suffix + ".pmml")

print(list(pipeline.predict(data[:10])))
print(list(pipeline.predict_proba(data[0:1])[0]))
