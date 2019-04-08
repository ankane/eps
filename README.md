# Eps

Linear regression for Ruby

- Build models quickly and easily
- Serve models built in Ruby, Python, R, and more
- Automatically handles categorical variables
- No external dependencies
- Works great with the SciRuby ecosystem (Daru & IRuby)

Check out [this post](https://ankane.org/rails-meet-data-science) for more info on data science with Rails

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
model = Eps::Regressor.new(data, target: :price)
puts model.summary
```

Make a prediction

```ruby
model.predict(bedrooms: 2, bathrooms: 1)
```

> Pass an array of hashes make multiple predictions at once

## Building Models

### Training and Test Sets

When building models, it’s a good idea to hold out some data so you can see how well the model will perform on unseen data. To do this, we split our data into two sets: training and test. We build the model with the training set and later evaluate it on the test set.

```ruby
split_date = Date.parse("2018-06-01")
train_set, test_set = houses.partition { |h| h.sold_at < split_date }
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

Selecting features for a model is extremely important for performance. Features can be numeric or categorical. For categorical features, there’s no need to create dummy variables - just pass the data as strings.

```ruby
{state: "CA"}
```

> Categorical features generate coefficients for each distinct value except for one

You should do this for any ids in your data.

```ruby
{city_id: "123"}
```

For times, create features like day of week and hour of day with:

```ruby
{weekday: time.wday.to_s, hour: time.hour.to_s}
```

In practice, your code may look like:

```ruby
def features(house)
  {
    bedrooms: house.bedrooms,
    city_id: house.city_id.to_s,
    month: house.sold_at.strftime("%b")
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
model = Eps::Regressor.new(train_features, train_target)
puts model.summary
```

The summary includes the coefficients and their significance. The lower the p-value, the more significant the feature is. p-values below 0.05 are typically considered significant. It also shows the adjusted r-squared, which is a measure of how well the model fits the data. The higher the number, the better the fit. Here’s a good explanation of why it’s [better than r-squared](https://www.quora.com/What-is-the-difference-between-R-squared-and-Adjusted-R-squared).

### Evaluation

When you’re happy with the model, see how well it performs on the test set. This gives us an idea of how well it’ll perform on unseen data.

```ruby
test_features = test_set.map { |h| features(h) }
test_target = test_set.map { |h| target(h) }
model.evaluate(test_features, test_target)
```

This returns:

- RMSE - Root mean square error
- MAE - Mean absolute error
- ME - Mean error

We want to minimize the RMSE and MAE and keep the ME around 0.

### Finalize

Now that we have an idea of how the model will perform, we want to retrain the model with all of our data. Treat outliers and missing data the same as you did with the training set.

```ruby
# outliers and missing data
houses.reject! { |h| h.bedrooms.nil? || h.price < 10000 }

# training
all_features = houses.map { |h| features(h) }
all_target = houses.map { |h| target(h) }
model = Eps::Regressor.new(all_features, all_target)
```

We now have a model that’s ready to serve.

## Serving Models

Once the model is trained, all we need are the coefficients to make predictions. You can dump them as a Ruby object or JSON. For Ruby, use:

```ruby
model.dump
```

Then hardcode the result into your app.

```ruby
data = {:coefficients=>{:_intercept=>63500.0, :bedrooms=>26000.0, :bathrooms=>10000.0}}
model = Eps::Regressor.load(data)
```

Now we can use it to make predictions.

```ruby
model.predict(bedrooms: 2, bathrooms: 1)
```

Another option that works well is writing the model to file in your app.

```ruby
json = model.to_json
File.open("model.json", "w") { |f| f.write(json) }
```

To load it, use:

```ruby
json = File.read("model.json")
model = Eps::Regressor.load_json(json)
```

To continuously train models, we recommend [storing them in your database](#database-storage).

### Other Languages

Eps makes it easy to serve models from other languages. You can build models in R, Python, and others and serve them in Ruby without having to worry about how to deploy or run another language. Eps can load models in:

JSON

```ruby
data = File.read("model.json")
model = Eps::Regressor.load_json(data)
```

[PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language) - Predictive Model Markup Language

```ruby
data = File.read("model.pmml")
model = Eps::Regressor.load_pmml(data)
```

> Loading PMML requires Nokogiri to be installed

[PFA](http://dmg.org/pfa/) - Portable Format for Analytics

```ruby
data = File.read("model.pfa")
model = Eps::Regressor.load_pfa(data)
```

Here are examples for how to dump models in each:

- [R JSON](guides/Modeling.md#r-json)
- [R PMML](guides/Modeling.md#r-pmml)
- [R PFA](guides/Modeling.md#r-pfa)
- [Python JSON](guides/Modeling.md#python-json)
- [Python PMML](guides/Modeling.md#python-pmml)
- [Python PFA](guides/Modeling.md#python-pfa)

### Verifying

It’s important for features to be implemented consistently when serving models created in other languages. We highly recommend verifying this programmatically. Create a CSV file with ids and predictions from the original model.

house_id | prediction
--- | ---
1 | 145000
2 | 123000
3 | 250000

Once the model is implemented in Ruby, confirm the predictions match.

```ruby
model = Eps::Regressor.load_json("model.json")

# preload houses to prevent n+1
houses = House.all.index_by(&:id)

CSV.foreach("predictions.csv", headers: true) do |row|
  house = houses[row["house_id"].to_i]
  expected = row["prediction"].to_f

  actual = model.predict(bedrooms: house.bedrooms, bathrooms: house.bathrooms)

  unless (actual - expected).abs < 0.001
    raise "Bad prediction for house #{house.id} (exp: #{expected}, act: #{actual})"
  end

  putc "✓"
end
```

### Database Storage

The database is another place you can store models. It’s good if you retrain models automatically.

> We recommend adding monitoring and guardrails as well if you retrain automatically

Create an ActiveRecord model to store the predictive model.

```sh
rails g model Model key:string:uniq data:text
```

Store the model with:

```ruby
store = Model.where(key: "price").first_or_initialize
store.update(data: model.to_json)
```

Load the model with:

```ruby
data = Model.find_by!(key: "price").data
model = Eps::Regressor.load_json(data)
```

## Monitoring

We recommend monitoring how well your models perform over time. To do this, save your predictions to the database. Then, compare them with:

```ruby
actual = houses.map(&:price)
estimated = houses.map(&:estimated_price)
Eps.metrics(actual, estimated)
```

This returns the same evaluation metrics as model building. For RMSE and MAE, alert if they rise above a certain threshold. For ME, alert if it moves too far away from 0.

## Rails

In Rails, we recommend storing models in the `app/stats_models` directory. Be sure to restart Spring after creating the directory so files are autoloaded. Here’s what a complete model in `app/stats_models/price_model.rb` may look like:

```ruby
module PriceModel
  def build
    houses = House.all.to_a

    # divide into training and test set
    rng = Random.new(1)
    train_set, test_set = houses.partition { rng.rand < 0.7 }

    # handle outliers and missing values
    train_set = preprocess(train_set)

    # train
    train_features = train_set.map { |v| features(v) }
    train_target = train_set.map { |v| target(v) }
    model = Eps::Regressor.new(train_features, train_target)
    puts model.summary

    # evaluate
    test_features = test_set.map { |v| features(v) }
    test_target = test_set.map { |v| target(v) }
    metrics = model.evaluate(test_features, test_target)
    puts "Test RMSE: #{metrics[:rmse]}"

    # finalize
    houses = preprocess(houses)
    all_features = houses.map { |h| features(h) }
    all_target = houses.map { |h| target(h) }
    @model = Eps::Regressor.new(all_features, all_target)

    # save
    File.open(model_file, "w") { |f| f.write(@model.json) }
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
      month: house.sold_at.strftime("%b")
    }
  end

  def target(house)
    house.price
  end

  def model
    @model ||= Eps::Regressor.load_json(File.read(model_file))
  end

  def model_file
    Rails.root.join("app", "stats_models", "price_model.json")
  end

  extend self # make all methods class methods
end
```

Build the model with:

```ruby
PriceModel.build
```

This saves the model to `app/stats_models/price_model.json`. Be sure to check this into source control.

Predict with:

```ruby
PriceModel.predict(house)
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

## Data

A number of data formats are supported. You can pass the target variable separately.

```ruby
x = [{x: 1}, {x: 2}, {x: 3}]
y = [1, 2, 3]
Eps::Regressor.new(x, y)
```

Or pass arrays of arrays

```ruby
x = [[1, 2], [2, 0], [3, 1]]
y = [1, 2, 3]
Eps::Regressor.new(x, y)
```

## Daru

Eps works well with Daru data frames.

```ruby
df = Daru::DataFrame.from_csv("houses.csv")
Eps::Regressor.new(df, target: "price")
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

Get coefficients

```ruby
model.coefficients
```

Get an extended summary with standard error, t-values, and r-squared

```ruby
model.summary(extended: true)
```

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
