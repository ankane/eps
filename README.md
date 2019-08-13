# Eps

Machine learning for Ruby

- Build predictive models quickly and easily
- Serve models built in Ruby, Python, R, and more
- No prior knowledge of machine learning required :tada:

Check out [this post](https://ankane.org/rails-meet-data-science) for more info on machine learning with Rails

[![Build Status](https://travis-ci.org/ankane/eps.svg?branch=master)](https://travis-ci.org/ankane/eps)

## Installation

Add this line to your application’s Gemfile:

```ruby
gem 'eps'
```

To speed up training on large datasets, you can also [add GSL](#training-performance).

## Getting Started

Create a model

```ruby
data = [
  {bedrooms: 1, bathrooms: 1, price: 100000},
  {bedrooms: 2, bathrooms: 1, price: 125000},
  {bedrooms: 2, bathrooms: 2, price: 135000},
  {bedrooms: 3, bathrooms: 2, price: 162000}
]
model = Eps::Model.new(data, target: :price)
puts model.summary
```

Make a prediction

```ruby
model.predict(bedrooms: 2, bathrooms: 1)
```

> Pass an array of hashes make multiple predictions at once

The target can be numeric (regression) or categorical (classification). Eps (short for “epsilon”) uses linear regression for regression and naive Bayes for classification.

## Building Models

### Training and Test Sets

When building models, it’s a good idea to hold out some data so you can see how well the model will perform on unseen data. To do this, we split our data into two sets: training and test. We build the model with the training set and later evaluate it on the test set.

```ruby
houses = House.all
split_date = Date.parse("2018-06-01")
train_set, test_set = houses.partition { |h| h.listed_at < split_date }
```

If your data doesn’t have a time associated with it, you can split it randomly.

```ruby
rng = Random.new(1) # seed random number generator
train_set, test_set = houses.partition { rng.rand < 0.7 }
```

### Outliers and Missing Data

Next, decide what to do with outliers and missing data. There are a number of methods for handling them, but the easiest is to remove them.

```ruby
train_set.reject! { |h| h.bedrooms.nil? || h.price < 10000 }
```

### Feature Engineering

Selecting features for a model is extremely important for performance. Features can be:

1. numeric
2. categorical

#### Numeric

For numeric features, use any numeric type.

```ruby
{bedrooms: 4, bathrooms: 2.5}
```

#### Categorical

For categorical features, use strings.

```ruby
{state: "CA"}
```

> Categorical features generate coefficients for each distinct value except for one

Convert any ids to strings so they’re treated as categorical features.

```ruby
{city_id: city_id.to_s}
```

For dates, create features like day of week and month.

```ruby
{weekday: sold_on.strftime("%a"), month: sold_on.strftime("%b")}
```

For times, create features like day of week and hour of day.

```ruby
{weekday: listed_at.strftime("%a"), hour: listed_at.hour.to_s}
```

---

In practice, your code may look like:

```ruby
def features(house)
  {
    bedrooms: house.bedrooms,
    city_id: house.city_id.to_s,
    month: house.listed_at.strftime("%b")
  }
end

train_features = train_set.map { |h| features(h) }
```

> We use a method for features so it can be used across training, evaluation, and prediction

We also need to prepare the target variable.

```ruby
def target(house)
  house.price
end

train_target = train_set.map { |h| target(h) }
```

### Training

Now, let’s train the model.

```ruby
model = Eps::Model.new(train_features, train_target)
puts model.summary
```

For regression, the summary includes the coefficients and their significance. The lower the p-value, the more significant the feature is. p-values below 0.05 are typically considered significant. It also shows the adjusted r-squared, which is a measure of how well the model fits the data. The higher the number, the better the fit. Here’s a good explanation of why it’s [better than r-squared](https://www.quora.com/What-is-the-difference-between-R-squared-and-Adjusted-R-squared).

### Evaluation

When you’re happy with the model, see how well it performs on the test set. This gives us an idea of how well it’ll perform on unseen data.

```ruby
test_features = test_set.map { |h| features(h) }
test_target = test_set.map { |h| target(h) }
puts model.evaluate(test_features, test_target)
```

For regression, this returns:

- RMSE - Root mean square error
- MAE - Mean absolute error
- ME - Mean error

We want to minimize the RMSE and MAE and keep the ME around 0.

For classification, this returns:

- Accuracy

We want to maximize the accuracy.

### Finalize

Now that we have an idea of how the model will perform, we want to retrain the model with all of our data. Treat outliers and missing data the same as you did with the training set.

```ruby
# outliers and missing data
houses.reject! { |h| h.bedrooms.nil? || h.price < 10000 }

# training
all_features = houses.map { |h| features(h) }
all_target = houses.map { |h| target(h) }
model = Eps::Model.new(all_features, all_target)
```

We now have a model that’s ready to serve.

## Serving Models

Once the model is trained, we need to store it. Eps uses PMML - [Predictive Model Markup Language](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language) - a standard for storing models. A great option is to write the model to a file with:

```ruby
File.write("model.pmml", model.to_pmml)
```

> You may need to add `nokogiri` to your Gemfile

To load a model, use:

```ruby
pmml = File.read("model.pmml")
model = Eps::Model.load_pmml(pmml)
```

Now we can use it to make predictions.

```ruby
model.predict(bedrooms: 2, bathrooms: 1)
```

To continuously train models, we recommend [storing them in your database](#database-storage).

## Full Example

We recommend putting all the model code in a single file. This makes it easy to rebuild the model as needed.

In Rails, we recommend creating a `app/ml_models` directory. Be sure to restart Spring after creating the directory so files are autoloaded.

```sh
bin/spring stop
```

Here’s what a complete model in `app/ml_models/price_model.rb` may look like:

```ruby
class PriceModel < Eps::Base
  def build
    houses = House.all.to_a

    # divide into training and test set
    split_date = Date.parse("2018-06-01")
    train_set, test_set = houses.partition { |h| h.listed_at < split_date }

    # handle outliers and missing values
    train_set = preprocess(train_set)

    # train
    train_features = train_set.map { |v| features(v) }
    train_target = train_set.map { |v| target(v) }
    model = Eps::Model.new(train_features, train_target)
    puts model.summary

    # evaluate
    test_features = test_set.map { |v| features(v) }
    test_target = test_set.map { |v| target(v) }
    metrics = model.evaluate(test_features, test_target)
    puts "Test RMSE: #{metrics[:rmse]}"
    # for classification, use:
    # puts "Test accuracy: #{(100 * metrics[:accuracy]).round}%"

    # finalize
    houses = preprocess(houses)
    all_features = houses.map { |h| features(h) }
    all_target = houses.map { |h| target(h) }
    model = Eps::Model.new(all_features, all_target)

    # save
    File.write(model_file, model.to_pmml)
    @model = nil # reset for future predictions
  end

  def predict(house)
    model.predict(features(house))
  end

  private

  def preprocess(train_set)
    train_set.reject { |h| h.bedrooms.nil? || h.price < 10000 }
  end

  def features(house)
    {
      bedrooms: house.bedrooms,
      city_id: house.city_id.to_s,
      month: house.listed_at.strftime("%b")
    }
  end

  def target(house)
    house.price
  end

  def model
    @model ||= Eps::Model.load_pmml(File.read(model_file))
  end

  def model_file
    File.join(__dir__, "price_model.pmml")
  end
end
```

Build the model with:

```ruby
PriceModel.build
```

This saves the model to `price_model.pmml`. Be sure to check this into source control.

Predict with:

```ruby
PriceModel.predict(house)
```

## Monitoring

We recommend monitoring how well your models perform over time. To do this, save your predictions to the database. Then, compare them with:

```ruby
actual = houses.map(&:price)
predicted = houses.map(&:predicted_price)
Eps.metrics(actual, predicted)
```

This returns the same evaluation metrics as model building. For RMSE and MAE, alert if they rise above a certain threshold. For ME, alert if it moves too far away from 0. For accuracy, alert if it drops below a certain threshold.

## Other Languages

Eps makes it easy to serve models from other languages. You can build models in R, Python, and others and serve them in Ruby without having to worry about how to deploy or run another language.

Eps can serve linear regression and naive Bayes models. Check out [Scoruby](https://github.com/asafschers/scoruby) to serve other models.

### R

To create a model in R, install the [pmml](https://cran.r-project.org/package=pmml) package

```r
install.packages("pmml")
```

For regression, run:

```r
library(pmml)

model <- lm(dist ~ speed,  cars)

# save model
data <- toString(pmml(model))
write(data, file="model.pmml")
```

For classification, run:

```r
library(pmml)
library(e1071)

model <- naiveBayes(Species ~ .,  iris)

# save model
data <- toString(pmml(model, predictedField="Species"))
write(data, file="model.pmml")
```

### Python

To create a model in Python, install the [sklearn2pmml](https://github.com/jpmml/sklearn2pmml) package

```sh
pip install sklearn2pmml
```

For regression, run:

```python
from sklearn2pmml import sklearn2pmml, make_pmml_pipeline
from sklearn.linear_model import LinearRegression

x = [1, 2, 3, 5, 6]
y = [5 * xi + 3 for xi in x]

model = LinearRegression()
model.fit([[xi] for xi in x], y)

# save model
sklearn2pmml(make_pmml_pipeline(model), "model.pmml")
```

For classification, run:

```python
from sklearn2pmml import sklearn2pmml, make_pmml_pipeline
from sklearn.naive_bayes import GaussianNB

x = [1, 2, 3, 5, 6]
y = ["ham", "ham", "ham", "spam", "spam"]

model = GaussianNB()
model.fit([[xi] for xi in x], y)

sklearn2pmml(make_pmml_pipeline(model), "model.pmml")
```

### Verifying

It’s important for features to be implemented consistently when serving models created in other languages. We highly recommend verifying this programmatically. Create a CSV file with ids and predictions from the original model.

house_id | prediction
--- | ---
1 | 145000
2 | 123000
3 | 250000

Once the model is implemented in Ruby, confirm the predictions match.

```ruby
model = Eps::Model.load_pmml("model.pmml")

# preload houses to prevent n+1
houses = House.all.index_by(&:id)

CSV.foreach("predictions.csv", headers: true, converters: :numeric) do |row|
  house = houses[row["house_id"]]
  expected = row["prediction"]

  actual = model.predict(bedrooms: house.bedrooms, bathrooms: house.bathrooms)

  success = actual.is_a?(String) ? actual == expected : (actual - expected).abs < 0.001
  raise "Bad prediction for house #{house.id} (exp: #{expected}, act: #{actual})" unless success

  putc "✓"
end
```

## Database Storage

The database is another place you can store models. It’s good if you retrain models automatically.

> We recommend adding monitoring and guardrails as well if you retrain automatically

Create an ActiveRecord model to store the predictive model.

```sh
rails g model Model key:string:uniq data:text
```

Store the model with:

```ruby
store = Model.where(key: "price").first_or_initialize
store.update(data: model.to_pmml)
```

Load the model with:

```ruby
data = Model.find_by!(key: "price").data
model = Eps::Model.load_pmml(data)
```

## Training Performance

Speed up training on large datasets with GSL.

First, [install GSL](https://www.gnu.org/software/gsl/). With Homebrew, you can use:

```sh
brew install gsl
```

Then, add this line to your application’s Gemfile:

```ruby
gem 'gsl', group: :development
```

It only needs to be available in environments used to build the model.

> This only speeds up regression, not classification

## Data

A number of data formats are supported. You can pass the target variable separately.

```ruby
x = [{x: 1}, {x: 2}, {x: 3}]
y = [1, 2, 3]
Eps::Model.new(x, y)
```

Or pass arrays of arrays

```ruby
x = [[1, 2], [2, 0], [3, 1]]
y = [1, 2, 3]
Eps::Model.new(x, y)
```

## Daru

Eps works well with Daru data frames.

```ruby
df = Daru::DataFrame.from_csv("houses.csv")
Eps::Model.new(df, target: "price")
```

To split into training and test sets, use:

```ruby
rng = Random.new(1) # seed random number generator
train_index = houses.map { rng.rand < 0.7 }
train_set = houses.where(train_index)
test_set = houses.where(train_index.map { |v| !v })
```

## CSVs

When importing data from CSV files, be sure to convert numeric fields. The `table` method does this automatically.

```ruby
CSV.table("data.csv").map { |row| row.to_h }
```

## Jupyter & IRuby

You can use [IRuby](https://github.com/SciRuby/iruby) to run Eps in [Jupyter](https://jupyter.org/) notebooks. Here’s how to get [IRuby working with Rails](https://ankane.org/jupyter-rails).

## Reference

Get an extended summary with standard error, t-values, and r-squared

```ruby
model.summary(extended: true)
```

## Upgrading

## 0.2.0

Eps 0.2.0 brings a number of improvements, including support for classification.

We recommend:

1. Changing `Eps::Regressor` to `Eps::Model`
2. Converting models from JSON to PMML

  ```ruby
  model = Eps::Model.load_json("model.json")
  File.write("model.pmml", model.to_pmml)
  ```

3. Renaming `app/stats_models` to `app/ml_models`

## History

View the [changelog](https://github.com/ankane/eps/blob/master/CHANGELOG.md)

## Contributing

Everyone is encouraged to help improve this project. Here are a few ways you can help:

- [Report bugs](https://github.com/ankane/eps/issues)
- Fix bugs and [submit pull requests](https://github.com/ankane/eps/pulls)
- Write, clarify, or fix documentation
- Suggest or add new features

To get started with development and testing:

```sh
git clone https://github.com/ankane/eps.git
cd eps
bundle install
rake test
```
