require_relative "test_helper"

class LinearRegressionTest < Minitest::Test
  def test_mpg
    data = mpg_data
    model = Eps::LinearRegression.new(data, target: :hwy, split: false)
    assert model.summary

    expected = [30.08293, 30.34814, 18.87341, 16.16030, 30.29378, 30.67852, 29.40943, 16.95101, 24.09529, 30.67852]
    predictions = model.predict(data.first(10))
    assert_elements_in_delta expected, predictions

    model = Eps::Model.load_pmml(model.to_pmml)
    predictions = model.predict(data.first(10))
    assert_elements_in_delta expected, predictions
  end

  def test_weight
    data = mpg_data
    model = Eps::LinearRegression.new(data, target: :hwy, weight: :cyl, split: false)

    expected = [29.11587889, 29.72208626, 18.19186325, 14.85352963, 29.95274992, 30.16211498, 28.27000463, 17.99179601, 24.76579075, 30.16211498]
    predictions = model.predict(data.first(10))
    assert_elements_in_delta expected, predictions
  end

  # this option behaves the same as Python and R when all columns are numeric
  # R generates the same number of coefficients when "+ 0" added to formula
  # it replaces the intercept with a single categorical value that was previously excluded
  # scikit-learn only supports numeric columns
  def test_intercept_false
    data = mpg_data.map { |d| d.slice(:displ, :year, :cyl, :hwy) }
    model = Eps::LinearRegression.new(data, target: :hwy, split: false, intercept: false)
    assert model.summary

    expected = [28.19885, 29.01272, 22.16365, 15.48646, 28.98745, 28.98745, 28.02697, 18.44373, 24.10989, 28.98745]
    predictions = model.predict(data.first(10))
    assert_elements_in_delta expected, predictions

    model = Eps::Model.load_pmml(model.to_pmml)
    predictions = model.predict(data.first(10))
    assert_elements_in_delta expected, predictions
  end

  def test_python_pmml
    data = mpg_data
    model = Eps::Model.load_pmml(File.read("test/support/python/linear_regression.pmml"))

    expected = [30.08293, 30.34814, 18.87341, 16.16030, 30.29378, 30.67852, 29.40943, 16.95101, 24.09529, 30.67852]
    predictions = model.predict(data.first(10))
    assert_elements_in_delta expected, predictions
  end

  def test_r_pmml
    data = mpg_data
    model = Eps::Model.load_pmml(File.read("test/support/r/linear_regression.pmml"))

    expected = [30.08293, 30.34814, 18.87341, 16.16030, 30.29378, 30.67852, 29.40943, 16.95101, 24.09529, 30.67852]
    predictions = model.predict(data.first(10))
    assert_elements_in_delta expected, predictions
  end

  def test_mpg_text
    data = mpg_data(extra_fields: [:model])
    model = Eps::LinearRegression.new(data, target: :hwy, split: false, text_features: {model: {max_features: 5}})

    expected = [30.07070759, 30.47815975, 18.5845258, 16.16074285, 30.15577456, 30.64164446, 29.5552295, 16.47974398, 24.3411665, 30.64164446]
    predictions = model.predict(data.first(10))
    assert_elements_in_delta expected, predictions
  end

  def test_python_pmml_text
    data = mpg_data(extra_fields: [:model])
    model = Eps::Model.load_pmml(File.read("test/support/python/linear_regression_text.pmml"))

    expected = [30.07070759, 30.47815975, 18.5845258, 16.16074285, 30.15577456, 30.64164446, 29.5552295, 16.47974398, 24.3411665, 30.64164446]
    predictions = model.predict(data.first(10))
    assert_elements_in_delta expected, predictions
  end

  def test_simple_regression
    data = [1, 2, 3, 4, 5].map { |xi| {x: xi, y: 3 + xi * 5} }

    model = Eps::LinearRegression.new(data, target: :y, split: false)
    predictions = model.predict([{x: 6}, {x: 7}])
    coefficients = model.coefficients

    assert_in_delta 33, predictions[0]
    assert_in_delta 38, predictions[1]

    assert_in_delta 3, coefficients[:_intercept]
    assert_in_delta 5, coefficients[:x]

    model = Eps::Model.load_pmml(model.to_pmml)
    predictions = model.predict([{x: 6}, {x: 7}])
    coefficients = model.coefficients

    assert_in_delta 33, predictions[0]
    assert_in_delta 38, predictions[1]

    assert_in_delta 3, coefficients[:_intercept]
    assert_in_delta 5, coefficients[:x]
  end

  def test_multiple_regression
    x = [[1, 0], [2, 4], [3, 5], [4, 2], [5, 1]]
    data = x.map { |v| {x: v[0], x2: v[1], y: 3 + v[0] * 5 + v[1] * 8} }

    model = Eps::LinearRegression.new(data, target: :y, split: false)
    predictions = model.predict([{x: 6, x2: 3}, {x: 7, x2: 4}])
    coefficients = model.coefficients

    assert_in_delta 57, predictions[0]
    assert_in_delta 70, predictions[1]

    assert_in_delta 3, coefficients[:_intercept]
    assert_in_delta 5, coefficients[:x]
    assert_in_delta 8, coefficients[:x2]
  end

  def test_multiple_solutions
    data = [1, 2, 3, 4, 5].map { |xi| {x: xi, x2: xi, y: 3 + xi * 5} }

    model = Eps::LinearRegression.new(data, target: :y, split: false)
    coefficients = model.coefficients

    assert_in_delta 3, coefficients[:_intercept]

    if gsl?
      assert_in_delta 2.5, coefficients[:x]
      assert_in_delta 2.5, coefficients[:x2]
    else
      assert_in_delta 5, coefficients[:x]
      assert_in_delta 0, coefficients[:x2]
    end
  end

  def test_multiple_solutions_constant
    data = [1, 2, 3, 4, 5].map { |xi| {x: xi, x2: 1, y: 3 + xi * 5} }

    model = Eps::LinearRegression.new(data, target: :y, split: false)
    coefficients = model.coefficients

    assert_in_delta 5, coefficients[:x]
    if gsl?
      assert_in_delta 1.5, coefficients[:_intercept]
      assert_in_delta 1.5, coefficients[:x2]
    else
      assert_in_delta 3, coefficients[:_intercept]
      assert_in_delta 0, coefficients[:x2]
    end
  end

  def test_separate_target
    x = [1, 2, 3, 4, 5].map { |xi| {x: xi} }
    y = x.map { |xi| 3 + xi[:x] * 5 }

    model = Eps::LinearRegression.new(x, y, split: false)
    predictions = model.predict([{x: 6}, {x: 7}])
    coefficients = model.coefficients

    assert_in_delta 33, predictions[0]
    assert_in_delta 38, predictions[1]

    assert_in_delta 3, coefficients[:_intercept]
    assert_in_delta 5, coefficients[:x]
  end

  def test_simple_array
    x = [1, 2, 3, 4, 5]
    y = x.map { |xi| 3 + xi * 5 }

    model = Eps::LinearRegression.new(x, y, split: false)
    predictions = model.predict([6, 7])
    coefficients = model.coefficients

    assert_in_delta 33, predictions[0]
    assert_in_delta 38, predictions[1]

    assert_in_delta 3, coefficients[:_intercept]
    assert_in_delta 5, coefficients[:x0]
  end

  def test_array
    x = [[1], [2], [3], [4], [5]]
    y = x.map { |xi| 3 + xi[0] * 5 }

    model = Eps::LinearRegression.new(x, y, split: false)
    predictions = model.predict([[6], [7]])
    coefficients = model.coefficients

    assert_in_delta 33, predictions[0]
    assert_in_delta 38, predictions[1]

    assert_in_delta 3, coefficients[:_intercept]
    assert_in_delta 5, coefficients[:x0]
  end

  def test_evaluate
    data = [1, 2, 3, 4, 5].map { |xi| {x: xi, y: 3 + xi * 5} }

    model = Eps::LinearRegression.new(data, target: :y, split: false)
    metrics = model.evaluate([{x: 6, y: 33}, {x: 7, y: 36}])

    assert_in_delta 1.4142, metrics[:rmse]
    assert_in_delta 1, metrics[:mae]
    assert_in_delta -1, metrics[:me]
  end

  def test_categorical
    data = [
      {x: "Sunday", y: 3},
      {x: "Sunday", y: 3},
      {x: "Monday", y: 5},
      {x: "Monday", y: 5}
    ]

    model = Eps::LinearRegression.new(data, target: :y, split: false)
    coefficients = model.coefficients

    assert_in_delta 3, coefficients[:_intercept]
    assert_in_delta 2, coefficients[:xMonday]

    assert_includes model.summary, "adjusted r2:"
    assert_includes model.summary(extended: true), "stderr"
  end

  def test_boolean
    data = [
      {x: false, y: 3},
      {x: false, y: 3},
      {x: true, y: 5},
      {x: true, y: 5}
    ]

    model = Eps::LinearRegression.new(data, target: :y, split: false)
    assert_elements_in_delta [3, 5], model.predict([{x: false}, {x: true}])

    pmml = model.to_pmml
    assert_valid_pmml(pmml)

    model = Eps::Model.load_pmml(pmml)
    assert_elements_in_delta [3, 5], model.predict([{x: false}, {x: true}])
  end

  def test_both
    data = [
      {x: 1, weekday: "Sunday", y: 12},
      {x: 2, weekday: "Sunday", y: 14},
      {x: 3, weekday: "Monday", y: 22},
      {x: 4, weekday: "Monday", y: 24},
    ]

    model = Eps::LinearRegression.new(data, target: :y, split: false)
    predictions = model.predict([{x: 6, weekday: "Sunday"}, {x: 7, weekday: "Monday"}])
    coefficients = model.coefficients

    assert_in_delta 22, predictions[0]
    assert_in_delta 30, predictions[1]

    assert_in_delta 10, coefficients[:_intercept]
    assert_in_delta 2, coefficients[:x]
    assert_in_delta 6, coefficients[:weekdayMonday]
  end

  def test_unknown_target
    data = [1, 2, 3, 4, 5].map { |xi| {x: xi, y: 3 + xi * 5 + rand} }
    error = assert_raises do
      Eps::LinearRegression.new(data, target: :unknown)
    end
    assert_equal "Missing column: unknown", error.message
  end

  def test_missing_data
    data = [1, 2, 3, 4, 5].map { |xi| {x: xi, y: 3 + xi * 5 + rand} }
    data[3][:x] = nil
    error = assert_raises do
      Eps::LinearRegression.new(data, target: :y)
    end
    assert_equal "Missing values in column x", error.message
  end

  def test_predict_missing_extra_data
    data = [1, 2, 3, 4, 5].map { |xi| {x: xi, y: 3 + xi * 5} }
    model = Eps::LinearRegression.new(data, target: :y, split: false)
    predictions = model.predict([{x: 6, y: nil}, {x: 7, y: nil}])

    assert_in_delta 33, predictions[0]
    assert_in_delta 38, predictions[1]
  end

  def test_few_samples
    data = [
      {bedrooms: 1, bathrooms: 1, price: 100000},
      {bedrooms: 2, bathrooms: 1, price: 125000},
      {bedrooms: 2, bathrooms: 2, price: 135000}
    ]
    error = assert_raises do
      Eps::LinearRegression.new(data, target: :price)
    end
    assert_equal "Number of data points must be at least two more than number of features", error.message
  end

  def test_numo
    x = Numo::NArray.cast([[1], [2], [3], [4], [5]])
    y = Numo::NArray.cast(x[true, 0].map { |xi| 3 + xi * 5 })

    model = Eps::LinearRegression.new(x, y, split: false)
    predictions = model.predict(Numo::NArray.cast([[6], [7]]))
    coefficients = model.coefficients

    assert_in_delta 33, predictions[0]
    assert_in_delta 38, predictions[1]

    assert_in_delta 3, coefficients[:_intercept]
    assert_in_delta 5, coefficients[:x0]
  end

  def test_evaluate_numo
    x = Numo::NArray.cast([[1], [2], [3], [4], [5]])
    y = Numo::NArray.cast(x[true, 0].map { |xi| 3 + xi * 5 })

    model = Eps::LinearRegression.new(x, y, split: false)
    test_x = Numo::NArray.cast([[6], [7]])
    test_y = Numo::NArray.cast([33, 36])
    metrics = model.evaluate(test_x, test_y)

    assert_in_delta 1.4142, metrics[:rmse]
    assert_in_delta 1, metrics[:mae]
    assert_in_delta -1, metrics[:me]
  end

  def test_rover
    x = [1, 2, 3, 4, 5]
    y = x.map { |v| 3 + v * 5 }
    df = Rover::DataFrame.new({x: x, y: y})

    model = Eps::LinearRegression.new(df, target: :y, split: false)
    predictions = model.predict(Rover::DataFrame.new({x: [6, 7]}))
    coefficients = model.coefficients

    assert_in_delta 33, predictions[0]
    assert_in_delta 38, predictions[1]

    assert_in_delta 3, coefficients[:_intercept]
    assert_in_delta 5, coefficients[:x]
  end

  def test_evaluate_rover
    x = [1, 2, 3, 4, 5]
    y = x.map { |v| 3 + v * 5 }
    df = Rover::DataFrame.new({x: x, y: y})

    model = Eps::LinearRegression.new(df, target: :y, split: false)

    test_df = Rover::DataFrame.new({x: [6, 7], y: [33, 36]})
    metrics = model.evaluate(test_df)

    assert_in_delta 1.4142, metrics[:rmse]
    assert_in_delta 1, metrics[:mae]
    assert_in_delta -1, metrics[:me]
  end

  def test_daru
    x = [1, 2, 3, 4, 5]
    y = x.map { |v| 3 + v * 5 }
    df = Daru::DataFrame.new(x: x, y: y)

    model = Eps::LinearRegression.new(df, target: :y, split: false)
    predictions = model.predict(Daru::DataFrame.new(x: [6, 7]))
    coefficients = model.coefficients

    assert_in_delta 33, predictions[0]
    assert_in_delta 38, predictions[1]

    assert_in_delta 3, coefficients[:_intercept]
    assert_in_delta 5, coefficients[:x]
  end

  def test_evaluate_daru
    x = [1, 2, 3, 4, 5]
    y = x.map { |v| 3 + v * 5 }
    df = Daru::DataFrame.new(x: x, y: y)

    model = Eps::LinearRegression.new(df, target: :y, split: false)

    test_df = Daru::DataFrame.new(x: [6, 7], y: [33, 36])
    metrics = model.evaluate(test_df)

    assert_in_delta 1.4142, metrics[:rmse]
    assert_in_delta 1, metrics[:mae]
    assert_in_delta -1, metrics[:me]
  end

  def test_summary
    data = [1, 2, 3, 4, 5].map { |xi| {x: xi, y: 3 + xi * 5} }
    model = Eps::LinearRegression.new(data, target: :y, split: false)
    assert_match "coef", model.summary
  end

  def test_summary_extended
    data = [1, 2, 3, 4, 5].map { |xi| {x: xi, y: 3 + xi * 5} }
    model = Eps::LinearRegression.new(data, target: :y, split: false)
    assert_match "stderr", model.summary(extended: true)
  end

  def test_text_features
    data = [
      {x: "Sunday is the best", y: 3},
      {x: "Sunday is the best", y: 3},
      {x: "Monday is the best", y: 5},
      {x: "Monday is the best", y: 5}
    ]

    model = Eps::LinearRegression.new(data, target: :y, text_features: {x: {max_features: 1, max_occurrences: 2}})
    assert_elements_in_delta [3, 5], model.predict([{x: "Sunday is the best"}, {x: "Monday is the best"}])

    pmml = model.to_pmml
    assert_includes pmml, "RegressionModel"
    assert_valid_pmml(pmml)

    model = Eps::Model.load_pmml(pmml)
    assert_elements_in_delta [3, 5], model.predict([{x: "Sunday is the best!!"}, {x: "Monday is the best!!"}])
  end

  def test_text_features_case_sensitive
    data = [
      {x: "Sunday is the best", y: 3},
      {x: "Sunday is the best", y: 3},
      {x: "sunday is the best", y: 5},
      {x: "sunday is the best", y: 5}
    ]

    model = Eps::LinearRegression.new(data, target: :y, text_features: {x: {max_features: 1, max_occurrences: 2, case_sensitive: true}})
    assert_elements_in_delta [3, 5], model.predict([{x: "Sunday is the best"}, {x: "sunday is the best"}])

    model = Eps::Model.load_pmml(model.to_pmml)
    assert_elements_in_delta [3, 5], model.predict([{x: "Sunday is the best!!"}, {x: "sunday is the best!!"}])
  end

  def test_validation_set
    data = [1, 2, 3, 4, 5].map { |xi| {x: xi, y: 3 + xi * 5} }
    validation_set = [{x: 6, y: 33}, {x: 7, y: 41}]
    model = Eps::LinearRegression.new(data, target: :y, validation_set: validation_set)
  end

  # TODO better name
  def test_multiple_solutions2
    x = 10.times.map { |i| [i, i + 1] }
    y = x.map { |xi| xi[0] }
    if gsl?
      model = Eps::LinearRegression.new(x, y)
      assert model.summary
    else
      assert_raises do
        Eps::LinearRegression.new(x, y)
      end
    end
  end

  def test_unstable_solution
    x = 10.times.map { |i| [i ** 10, i ** 10 + 0.0001] }
    y = x.map { |xi| 1 }
    y[-1] = 2

    if gsl?
      model = Eps::LinearRegression.new(x, y)
      assert model.summary
    else
      assert_raises(Eps::UnstableSolution) do
        Eps::LinearRegression.new(x, y)
      end
    end
  end

  def test_many_rows
    data = 200000.times.map { |i| {x: i + rand, y: i * 2 } }
    Eps::LinearRegression.new(data, target: :y)
  end

  private

  def gsl?
    defined?(GSLR)
  end
end
