require_relative "test_helper"

class LightGBMTest < Minitest::Test
  def test_regression
    data = mpg_data(extra_fields: [:model])
    model = Eps::LightGBM.new(data, target: :hwy, split: false, text_features: {model: {max_features: 5}})
    assert model.summary

    expected = [30.65282474, 33.95768195, 17.89527884, 16.93797369, 28.38019179, 29.40602519, 27.6259532, 18.81081728, 24.88466865, 29.40602519]
    predictions = model.predict(data.first(10))
    assert_elements_in_delta expected, predictions

    model = Eps::Model.load_pmml(model.to_pmml)
    predictions = model.predict(data.first(10))
    assert_elements_in_delta expected, predictions
  end

  def test_regression_python_pmml
    data = mpg_data(extra_fields: [:model])
    model = Eps::Model.load_pmml(File.read("test/support/python/lightgbm_regression.pmml"))

    expected = [30.65282474, 33.95768195, 17.89527884, 16.93797369, 28.38019179, 29.40602519, 27.6259532, 18.81081728, 24.88466865, 29.40602519]
    predictions = model.predict(data.first(10))
    assert_elements_in_delta expected, predictions
  end

  def test_binary
    data = mpg_data(binary: true)
    model = Eps::LightGBM.new(data, target: :drv, split: false)

    expected = %w(f f 4 4 f f f 4 4 f)
    predictions = model.predict(data.first(10))
    assert_equal expected, predictions

    model = Eps::Model.load_pmml(model.to_pmml)
    predictions = model.predict(data.first(10))
    assert_equal expected, predictions
  end

  def test_binary_python_pmml
    data = mpg_data(binary: true)
    model = Eps::Model.load_pmml(File.read("test/support/python/lightgbm_binary.pmml"))

    expected = %w(f f 4 4 f f f 4 4 f)
    predictions = model.predict(data.first(10))
    assert_equal expected, predictions
  end

  def test_multiclass
    data = mpg_data
    data.each { |r| r.delete(:hwy) }
    model = Eps::LightGBM.new(data, target: :drv, split: false)

    expected = %w(f f 4 r f f f 4 4 f)
    predictions = model.predict(data.first(10))
    assert_equal expected, predictions

    model = Eps::Model.load_pmml(model.to_pmml)
    predictions = model.predict(data.first(10))
    assert_equal expected, predictions
  end

  def test_multiclass_python_pmml
    data = mpg_data
    model = Eps::Model.load_pmml(File.read("test/support/python/lightgbm_multiclass.pmml"))

    expected = %w(f f 4 r f f f 4 4 f)
    predictions = model.predict(data.first(10))
    assert_equal expected, predictions
  end

  def test_missing_column
    test_set = mpg_data(extra_fields: [:model]).first
    test_set.delete(:year)

    error = assert_raises(ArgumentError) do
      model.predict(test_set)
    end
    assert_equal "Missing column: year", error.message
  end

  def test_bad_type
    test_set = mpg_data(extra_fields: [:model]).first
    test_set[:year] = "bad"

    error = assert_raises(ArgumentError) do
      model.predict(test_set)
    end
    assert_equal "Bad type for column year: Expected numeric but got categorical", error.message
  end

  def test_text_features_regression
    data = [
      {x: "Sunday is the best", y: 3},
      {x: "Sunday is the day", y: 3},
      {x: "Monday is the best", y: 5},
      {x: "Monday is the day", y: 5}
    ]

    model = Eps::LightGBM.new(data, target: :y, early_stopping: false)
    assert_elements_in_delta [3, 5], model.predict([{x: "Sunday is the best"}, {x: "Monday is the best"}])

    pmml = model.to_pmml
    assert_includes pmml, "Segmentation"
    assert_valid_pmml(pmml)

    model = Eps::Model.load_pmml(pmml)
    assert_elements_in_delta [3, 5], model.predict([{x: "Sunday is the best!!"}, {x: "Monday is the best!!"}])
  end

  def test_text_features_regression_case_sensitive
    data = [
      {x: "Sunday is the best", y: 3},
      {x: "Sunday is the day", y: 3},
      {x: "sunday is the best", y: 5},
      {x: "sunday is the day", y: 5}
    ]

    model = Eps::LightGBM.new(data, target: :y, early_stopping: false, text_features: {x: {case_sensitive: true}})
    assert_elements_in_delta [3, 5], model.predict([{x: "Sunday is the best"}, {x: "sunday is the best"}])

    model = Eps::Model.load_pmml(model.to_pmml)
    assert_elements_in_delta [3, 5], model.predict([{x: "Sunday is the best!!"}, {x: "sunday is the best!!"}])
  end

  def test_text_features_classification
    data = [
      {message: "This is the first document.", tag: "ham"},
      {message: "Hello, this is the second document.", tag: "spam"},
      {message: "Hello, and this is the third one.", tag: "spam"},
      {message: "Is this the first document?", tag: "ham"}
    ]

    test_data = [
      {message: "what a cool first document"},
      {message: "hello document"}
    ]

    model = Eps::LightGBM.new(data, target: :tag)
    assert_equal ["ham", "spam"], model.predict(test_data)

    pmml = model.to_pmml
    assert_includes pmml, "Segmentation"
    assert_valid_pmml(pmml)

    model = Eps::Model.load_pmml(pmml)
    assert_equal ["ham", "spam"], model.predict(test_data)
  end

  private

  def model
    @model ||= Eps::Model.load_pmml(File.read("test/support/python/lightgbm_regression.pmml"))
  end
end
