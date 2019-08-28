require_relative "test_helper"

class RegressionTest < Minitest::Test
  def test_simple_regression
    data = [1, 2, 3, 4, 5].map { |xi| {x: xi, y: 3 + xi * 5} }

    model = Eps::Model.new(data, target: :y)
    predictions = model.predict([{x: 6}, {x: 7}])
    coefficients = model.coefficients

    assert_in_delta 33, predictions[0]
    assert_in_delta 38, predictions[1]

    assert_in_delta 3, coefficients[:_intercept]
    assert_in_delta 5, coefficients[:x]

    # model = Eps::Model.load_pmml(model.to_pmml)
    # predictions = model.predict([{x: 6}, {x: 7}])
    # coefficients = model.coefficients

    # assert_in_delta 33, predictions[0]
    # assert_in_delta 38, predictions[1]

    # assert_in_delta 3, coefficients[:_intercept]
    # assert_in_delta 5, coefficients[:x]

    byte_str = File.binread("test/support/linear_regression/houses.onnx")
    pp Eps::Utils.read_onnx(byte_str)
    require "onnxruntime"
    m = OnnxRuntime::Model.new(byte_str)
    p m.inputs
    p m.outputs
    pp Eps::Utils.read_onnx(model.to_onnx)

    model = Eps::Model.load_onnx(model.to_onnx)
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

    model = Eps::Model.new(data, target: :y)
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

    model = Eps::Model.new(data, target: :y)
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

    model = Eps::Model.new(data, target: :y)
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

    model = Eps::Model.new(x, y)
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

    model = Eps::Model.new(x, y)
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

    model = Eps::Model.new(x, y)
    predictions = model.predict([[6], [7]])
    coefficients = model.coefficients

    assert_in_delta 33, predictions[0]
    assert_in_delta 38, predictions[1]

    assert_in_delta 3, coefficients[:_intercept]
    assert_in_delta 5, coefficients[:x0]
  end

  def test_evaluate
    data = [1, 2, 3, 4, 5].map { |xi| {x: xi, y: 3 + xi * 5} }

    model = Eps::Model.new(data, target: :y)
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

    model = Eps::Model.new(data, target: :y)
    coefficients = model.coefficients

    assert_in_delta 3, coefficients[:_intercept]
    assert_in_delta 2, coefficients[:xMonday]

    assert_includes model.summary, "adjusted r2:"
    assert_includes model.summary(extended: true), "stderr"
  end

  def test_boolean
    data = [
      {x: true, y: 3},
      {x: true, y: 3},
      {x: false, y: 5},
      {x: false, y: 5}
    ]

    model = Eps::Model.new(data, target: :y)
    coefficients = model.coefficients

    assert_in_delta 3, coefficients[:_intercept]
    assert_in_delta 2, coefficients[:xfalse]
  end

  def test_both
    data = [
      {x: 1, weekday: "Sunday", y: 12},
      {x: 2, weekday: "Sunday", y: 14},
      {x: 3, weekday: "Monday", y: 22},
      {x: 4, weekday: "Monday", y: 24},
    ]

    model = Eps::Model.new(data, target: :y)
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
      Eps::Model.new(data, target: :unknown)
    end
    assert_equal "Missing column: unknown", error.message
  end

  def test_missing_data
    data = [1, 2, 3, 4, 5].map { |xi| {x: xi, y: 3 + xi * 5 + rand} }
    data[3][:x] = nil
    error = assert_raises do
      Eps::Model.new(data, target: :y)
    end
    assert_equal "Missing values in column x", error.message
  end

  def test_predict_missing_extra_data
    data = [1, 2, 3, 4, 5].map { |xi| {x: xi, y: 3 + xi * 5} }
    model = Eps::Model.new(data, target: :y)
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
      Eps::Model.new(data, target: :price)
    end
    assert_equal "Number of samples must be at least two more than number of features", error.message
  end

  def test_load
    model = Eps::Model.load(coefficients: {_intercept: 3, x: 5})
    predictions = model.predict([{x: 6}, {x: 7}])
    assert_in_delta 33, predictions[0]
    assert_in_delta 38, predictions[1]
  end

  def test_load_json
    data = File.read("test/support/linear_regression/model.json")
    model = Eps::Model.load_json(data)
    predictions = model.predict([{x: 6}, {x: 7}])
    assert_in_delta 33, predictions[0]
    assert_in_delta 38, predictions[1]
  end

  def test_load_json_python
    data = File.read("test/support/linear_regression/pymodel.json")
    model = Eps::Model.load_json(data)
    predictions = model.predict([{x: 6}, {x: 7}])
    assert_in_delta 33, predictions[0]
    assert_in_delta 38, predictions[1]
  end

  def test_load_parsed_json
    model = Eps::Model.load_json({"coefficients" => {"_intercept" => 3, "x" => 5}})
    predictions = model.predict([{x: 6}, {x: 7}])
    assert_in_delta 33, predictions[0]
    assert_in_delta 38, predictions[1]
  end

  def test_load_pmml
    data = File.read("test/support/linear_regression/model.pmml")
    model = Eps::Model.load_pmml(data)
    predictions = model.predict([{x: 6}, {x: 7}])
    assert_in_delta 33, predictions[0]
    assert_in_delta 38, predictions[1]
  end

  def test_load_pmml_string_keys
    data = File.read("test/support/linear_regression/model.pmml")
    model = Eps::Model.load_pmml(data)
    predictions = model.predict([{"x" => 6}, {"x" => 7}])
    assert_in_delta 33, predictions[0]
    assert_in_delta 38, predictions[1]
  end

  def test_load_pmml_categorical
    data = File.read("test/support/linear_regression/modelcat.pmml")
    model = Eps::Model.load_pmml(data)
    predictions = model.predict([{x: 6, weekday: "Sunday"}, {x: 7, weekday: "Monday"}])
    assert_in_delta 22, predictions[0]
    assert_in_delta 30, predictions[1]
  end

  def test_load_pmml_python
    data = File.read("test/support/linear_regression/pymodel.pmml")
    model = Eps::Model.load_pmml(data)
    predictions = model.predict([{x0: 6}, {x0: 7}])
    assert_in_delta 33, predictions[0]
    assert_in_delta 38, predictions[1]
  end

  def test_load_pfa
    data = File.read("test/support/linear_regression/model.pfa")
    model = Eps::Model.load_pfa(data)
    predictions = model.predict([{x: 6}, {x: 7}])
    assert_in_delta 33, predictions[0]
    assert_in_delta 38, predictions[1]
  end

  def test_load_pfa_categorical
    data = File.read("test/support/linear_regression/modelcat.pfa")
    model = Eps::Model.load_pfa(data)
    predictions = model.predict([{x: 6, weekday: "Sunday"}, {x: 7, weekday: "Monday"}])
    assert_in_delta 22, predictions[0]
    assert_in_delta 30, predictions[1]
  end

  def test_load_pfa_python
    data = File.read("test/support/linear_regression/pymodel.pfa")
    model = Eps::Model.load_pfa(data)
    predictions = model.predict([{x0: 6}, {x0: 7}])
    assert_in_delta 33, predictions[0]
    assert_in_delta 38, predictions[1]
  end

  def test_to_json
    data = File.read("test/support/linear_regression/modelcat.pmml")
    model = Eps::Model.load_pmml(data)
    assert_includes model.to_json, "weekdaySunday"
    assert_includes JSON.generate(model), "weekdaySunday"
  end

  def test_to_pmml
    data = File.read("test/support/linear_regression/modelcat.pmml")
    model = Eps::Model.load_pmml(data)
    pmml = model.to_pmml
    assert_includes pmml, "RegressionModel"
  end

  def test_daru
    x = [1, 2, 3, 4, 5]
    y = x.map { |v| 3 + v * 5 }
    df = Daru::DataFrame.new(x: x, y: y)

    model = Eps::Model.new(df, target: :y)
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

    model = Eps::Model.new(df, target: :y)

    test_df = Daru::DataFrame.new(x: [6, 7], y: [33, 36])
    metrics = model.evaluate(test_df)

    assert_in_delta 1.4142, metrics[:rmse]
    assert_in_delta 1, metrics[:mae]
    assert_in_delta -1, metrics[:me]
  end

  def test_summary
    data = [1, 2, 3, 4, 5].map { |xi| {x: xi, y: 3 + xi * 5} }
    model = Eps::Model.new(data, target: :y)
    assert_match "coef", model.summary
  end

  def test_summary_extended
    data = [1, 2, 3, 4, 5].map { |xi| {x: xi, y: 3 + xi * 5} }
    model = Eps::Model.new(data, target: :y)
    assert_match "stderr", model.summary(extended: true)
  end

  def test_metrics
    actual = [1, 2, 3]
    estimated = [1, 2, 9]
    metrics = Eps::Model.metrics(actual, estimated)

    assert_in_delta 3.464, metrics[:rmse]
    assert_in_delta 2, metrics[:mae]
    assert_in_delta -2, metrics[:me]
  end

  def test_validation_set
    data = [1, 2, 3, 4, 5].map { |xi| {x: xi, y: 3 + xi * 5} }
    validation_set = [{x: 6, y: 33}, {x: 7, y: 41}]

    model = Eps::Model.new
    model.train(data, target: :y, validation_set: validation_set)
  end

  private

  def gsl?
    ENV["GSL"]
  end
end
