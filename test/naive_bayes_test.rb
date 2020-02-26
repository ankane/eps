require_relative "test_helper"

class NaiveBayesTest < Minitest::Test
  def test_mpg
    data = mpg_data
    data = data.map { |r| r.slice(:displ, :year, :cyl, :class, :drv) }
    model = Eps::NaiveBayes.new(data, target: :drv, split: false)
    assert model.summary

    expected = %w(f f 4 r f f f 4 f f)
    predictions = model.predict(data.first(10))
    assert_equal expected, predictions

    model = Eps::Model.load_pmml(model.to_pmml)
    predictions = model.predict(data.first(10))
    assert_equal expected, predictions
  end

  def test_weight
    data = mpg_data
    assert_raises ArgumentError do
      Eps::NaiveBayes.new(data, target: :drv, weight: :cyl, split: false)
    end
  end

  def test_python_pmml
    data = mpg_data
    model = Eps::Model.load_pmml(File.read("test/support/python/naive_bayes.pmml"))

    # different expected than Ruby and R
    expected = %w(f f r r f f f r f f)
    predictions = model.predict(data.first(10))
    assert_equal expected, predictions
  end

  def test_r_pmml
    data = mpg_data
    model = Eps::Model.load_pmml(File.read("test/support/r/naive_bayes.pmml"))

    expected = %w(f f 4 r f f f 4 f f)
    predictions = model.predict(data.first(10))
    assert_equal expected, predictions
  end

  def test_bad_mpg
    data = mpg_data
    model = Eps::Model.load_pmml(File.read("test/support/bad_naive_bayes.pmml"))

    expected = %w(f f 4 r f f f 4 f f)
    predictions = model.predict(data.first(10))
    assert_equal expected, predictions

    model = Eps::Model.load_pmml(model.to_pmml)
    predictions = model.predict(data.first(10))
    assert_equal expected, predictions
  end

  def test_daru
    df = Daru::DataFrame.from_csv("test/support/mpg.csv")
    df = df["displ", "year", "cyl", "class", "drv"]
    df["drv"] = df["drv"].map(&:to_s)

    model = Eps::NaiveBayes.new(df, target: :drv, split: false)

    expected = %w(f f 4 r f f f 4 f f)
    predictions = model.predict(df.first(10))
    assert_equal expected, predictions
  end

  def test_unknown_categorical_value
    data = [
      {x: "Sunday", y: "red"},
      {x: "Sunday", y: "red"},
      {x: "Sunday", y: "red"},
      {x: "Monday", y: "blue"},
      {x: "Monday", y: "blue"},
    ]

    model = Eps::NaiveBayes.new(data, target: :y, split: false)
    assert_equal "red", model.predict(x: "Tuesday")
  end

  def test_text_features
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

    model = Eps::NaiveBayes.new(data, target: :tag, text_features: [:message])
    assert_equal ["ham", "spam"], model.predict(test_data)

    pmml = model.to_pmml
    assert_includes pmml, "NaiveBayesModel"
    assert_valid_pmml(pmml)

    model = Eps::Model.load_pmml(pmml)
    assert_equal ["ham", "spam"], model.predict(test_data)
  end

  def test_boolean
    data = [
      {x: false, y: "ham"},
      {x: false, y: "ham"},
      {x: true, y: "spam"},
      {x: true, y: "spam"}
    ]

    model = Eps::NaiveBayes.new(data, target: :y)
    assert_equal ["ham", "spam"], model.predict([{x: false}, {x: true}])

    pmml = model.to_pmml
    assert_valid_pmml(pmml)

    model = Eps::Model.load_pmml(pmml)
    assert_equal ["ham", "spam"], model.predict([{x: false}, {x: true}])
  end

  def test_probabilities
    data = mpg_data
    data = data.map { |r| r.slice(:displ, :year, :cyl, :class, :drv) }
    model = Eps::NaiveBayes.new(data, target: :drv, split: false)
    assert model.summary

    predictions = model.predict_probability(data.first)
    assert_in_delta 0.007528105, predictions["4"]
    assert_in_delta 9.924719e-01, predictions["f"]
    assert_in_delta 6.506897e-10, predictions["r"]

    model = Eps::Model.load_pmml(model.to_pmml)
    predictions = model.predict_probability(data.first)
    assert_in_delta 0.007528105, predictions["4"]
    assert_in_delta 9.924719e-01, predictions["f"]
    assert_in_delta 6.506897e-10, predictions["r"]
  end
end
