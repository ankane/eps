require_relative "test_helper"

class ModelTest < Minitest::Test
  def test_example
    data = [
      {bedrooms: 1, bathrooms: 1, price: 100000},
      {bedrooms: 2, bathrooms: 1, price: 125000},
      {bedrooms: 2, bathrooms: 2, price: 135000},
      {bedrooms: 3, bathrooms: 2, price: 162000}
    ]
    model = Eps::Model.new(data, target: :price)
    assert_in_delta 125000, model.predict(bedrooms: 2, bathrooms: 1)
  end

  def test_untrained
    model = Eps::Model.new

    # TODO better error
    assert_raises(NoMethodError) do
      model.predict([{x: 1}])
    end
  end

  def test_bad_pmml
    assert_raises(Nokogiri::XML::SyntaxError) do
      Eps::Model.load_pmml("bad")
    end
  end

  def test_unknown_target
    data = [1, 2, 3, 4, 5].map { |xi| {x: xi, y: 3 + xi * 5 + rand} }
    error = assert_raises do
      Eps::Model.new(data, target: :unknown)
    end
    assert_equal "Missing column: unknown", error.message
  end

  def test_boolean_target
    data = [
      {x: "Sunday", y: true},
      {x: "Sunday", y: true},
      {x: "Monday", y: false},
      {x: "Monday", y: false}
    ]

    # treat boolean values as strings
    model = Eps::Model.new(data, target: :y)
    assert_equal "false", model.predict(x: "Monday")

    model = Eps::Model.load_pmml(model.to_pmml)
    assert_equal "false", model.predict(x: "Monday")
  end

  def test_split
    model = Eps::Model.new(mpg_data, target: :hwy, split: true)
  end

  def test_split_symbol
    data = mpg_data.map { |r| r.merge(listed_at: Date.today - rand(10)) }
    model = Eps::Model.new(data, target: :hwy, split: :listed_at)
  end

  def test_split_shuffle_false
    data = mpg_data
    model = Eps::Model.new(data, target: :hwy, split: {shuffle: false})
  end

  def test_split_column
    data = mpg_data.map { |r| r.merge(listed_at: Date.today - rand(10)) }
    model = Eps::Model.new(data, target: :hwy, split: {column: :listed_at})
  end

  def test_split_column_value
    data = mpg_data.map { |r| r.merge(listed_at: Date.today - rand(10)) }
    model = Eps::Model.new(data, target: :hwy, split: {column: :listed_at, value: Date.today - 5})
  end

  def test_split_no_training_data
    data = mpg_data.map { |r| r.merge(listed_at: Date.today - rand(10)) }
    error = assert_raises do
      Eps::Model.new(data, target: :hwy, split: {column: :listed_at, value: Date.today - 20})
    end
    assert_equal "No data in training set", error.message
  end

  def test_split_no_validation_data
    data = mpg_data.map { |r| r.merge(listed_at: Date.today - rand(10)) }
    error = assert_raises do
      Eps::Model.new(data, target: :hwy, split: {column: :listed_at, value: Date.today + 20})
    end
    assert_equal "No data in validation set", error.message
  end

  def test_regression_comparison
    lr = Eps::LinearRegression.new(mpg_data, target: :hwy)
    lgb = Eps::LightGBM.new(mpg_data, target: :hwy)
    # puts lr.summary
    # puts lgb.summary
  end

  def test_classification_comparison
    nb = Eps::NaiveBayes.new(mpg_data, target: :drv)
    lgb = Eps::LightGBM.new(mpg_data, target: :drv)
    # puts nb.summary
    # puts lgb.summary
  end
end
