import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from lightgbm import LGBMRegressor
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.decoration import ContinuousDomain, CategoricalDomain
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml.preprocessing import PMMLLabelEncoder
from sklearn2pmml.feature_extraction.text import Splitter
from sklearn_pandas import DataFrameMapper

data = pd.read_csv("test/support/mpg.csv")

numeric_features = ["displ", "year", "cyl"]
categorical_features = ["drv", "class"]
text_features = ["model"]

mapper = DataFrameMapper(
  [(numeric_features, [ContinuousDomain()])] +
  [([f], [CategoricalDomain(), PMMLLabelEncoder()]) for f in categorical_features] +
  [(f, [CategoricalDomain(), CountVectorizer(tokenizer=Splitter(), max_features=5)]) for f in text_features]
)

pipeline = PMMLPipeline([
  ("mapper", mapper),
  ("model", LGBMRegressor(n_estimators=1000))
])
# use model__sample_weight for weight
pipeline.fit(data, data["hwy"], model__categorical_feature=[3, 4])

sklearn2pmml(pipeline, "test/support/python/lightgbm_regression.pmml")

print(pipeline.predict(data[:10]))
