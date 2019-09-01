require_relative "test_helper"

class ClassificationTest < Minitest::Test
  def test_simple_classification
    data = houses_data
    model = Eps::Model.new(data.map { |r| r.slice(:bedrooms, :bathrooms, :state, :color) }, target: :color)
    predictions = model.predict(data)
    correct = data.map { |r| r[:predicted_color] }
    errors = correct.select.with_index { |v, i| v != predictions[i] }
    assert_empty errors

    assert_includes model.summary, "accuracy:"

    model = Eps::Model.load_pmml(model.to_pmml)
    predictions = model.predict(data)
    correct = data.map { |r| r[:predicted_color] }
    errors = correct.select.with_index { |v, i| v != predictions[i] }
    assert_empty errors
  end

  def test_daru
    df = Daru::DataFrame.from_csv("test/support/data/houses.csv")
    df = df["bedrooms", "bathrooms", "state", "color"]

    model = Eps::Model.new(df, target: "color")
    model.predict(df)
  end

  # this is where smoothing is needed
  def test_unknown_categorical_value
    data = [
      {x: "Sunday", y: "red"},
      {x: "Sunday", y: "red"},
      {x: "Sunday", y: "red"},
      {x: "Monday", y: "blue"},
      {x: "Monday", y: "blue"},
    ]

    model = Eps::Model.new(data, target: :y)
    assert_equal "red", model.predict(x: "Tuesday")
  end

  def test_load_pmml
    data = File.read("test/support/naive_bayes/classifier.pmml")
    Eps::Model.load_pmml(data)
  end

  def test_to_pmml
    data = File.read("test/support/naive_bayes/classifier.pmml")
    model = Eps::Model.load_pmml(data)
    pmml = model.to_pmml
    assert_includes pmml, "NaiveBayesModel"
  end

  def test_example
    data = [
      {bedrooms: 1, bathrooms: 1, price: 100000, tag: "Low"},
      {bedrooms: 2, bathrooms: 1, price: 125000, tag: "Med"},
      {bedrooms: 2, bathrooms: 2, price: 135000, tag: "Med"},
      {bedrooms: 3, bathrooms: 2, price: 162000, tag: "High"}
    ]
    Eps::Model.new(data, target: :tag)
  end
end
