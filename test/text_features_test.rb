require_relative "test_helper"

class TextFeaturesTest < Minitest::Test
  def test_classification
    skip

     data = [
      {message: "This is the first document.", tag: "ham"},
      {message: "Hello, this document is the second document.", tag: "spam"},
      {message: "Hello, and this is the third one.", tag: "spam"},
      {message: "Is this the first document?", tag: "ham"}
    ]

    test_data = [
      {message: "what a cool first document"},
      {message: "hello document"}
    ]

    model = Eps::Model.new
    model.train(data, target: :tag, text_features: [:message])
    assert_equal ["ham", "spam"], model.predict(test_data)

    pmml = model.to_pmml
    assert_includes pmml, "NaiveBayesModel"
    assert_valid_pmml(pmml)

    model = Eps::Model.load_pmml(pmml)
    assert_equal ["ham", "spam"], model.predict(test_data)
  end

  def test_regression
    data = [
      {x: "Sunday is the best", y: 3},
      {x: "Sunday is the best", y: 3},
      {x: "Monday is the best", y: 5},
      {x: "Monday is the best", y: 5}
    ]

    model = Eps::Model.new
    model.train(data, target: :y, text_features: {x: {max_features: 1, max_occurrences: 2}})
    assert_equal [3, 5], model.predict([{x: "Sunday is the best"}, {x: "Monday is the best"}])

    pmml = model.to_pmml
    assert_includes pmml, "RegressionModel"
    assert_valid_pmml(pmml)

    model = Eps::Model.load_pmml(pmml)
    assert_equal [3, 5], model.predict([{x: "Sunday is the best!!"}, {x: "Monday is the best!!"}])
  end

  def test_case_sensitive
    data = [
      {x: "Sunday is the best", y: 3},
      {x: "Sunday is the best", y: 3},
      {x: "sunday is the best", y: 5},
      {x: "sunday is the best", y: 5}
    ]

    model = Eps::Model.new
    model.train(data, target: :y, text_features: {x: {max_features: 1, max_occurrences: 2, case_sensitive: true}})
    assert_equal [3, 5], model.predict([{x: "Sunday is the best"}, {x: "sunday is the best"}])

    model = Eps::Model.load_pmml(model.to_pmml)
    assert_equal [3, 5], model.predict([{x: "Sunday is the best!!"}, {x: "sunday is the best!!"}])
  end
end
