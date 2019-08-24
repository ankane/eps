require_relative "test_helper"

class ModelTest < Minitest::Test
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
    data = houses_data.map { |r| r.slice(:bedrooms, :bathrooms, :price) }
    model = Eps::Model.new
    model.train(data, target: :price, split: true)
  end

  def test_split_symbol
    data = houses_data.map { |r| r.slice(:bedrooms, :bathrooms, :price).merge(listed_at: Date.today - rand(10)) }
    model = Eps::Model.new
    model.train(data, target: :price, split: :listed_at)
  end

  def test_split_shuffle_false
    data = houses_data.map { |r| r.slice(:bedrooms, :bathrooms, :price) }
    model = Eps::Model.new
    model.train(data, target: :price, split: {shuffle: false})
  end

  def test_split_column
    data = houses_data.map { |r| r.slice(:bedrooms, :bathrooms, :price).merge(listed_at: Date.today - rand(10)) }
    model = Eps::Model.new
    model.train(data, target: :price, split: {column: :listed_at})
  end

  def test_split_column_value
    data = houses_data.map { |r| r.slice(:bedrooms, :bathrooms, :price).merge(listed_at: Date.today - rand(10)) }
    model = Eps::Model.new
    model.train(data, target: :price, split: {column: :listed_at, value: Date.today - 5})
  end

  def test_split_no_training_data
    data = houses_data.map { |r| r.slice(:bedrooms, :bathrooms, :price).merge(listed_at: Date.today - rand(10)) }
    model = Eps::Model.new
    error = assert_raises do
      model.train(data, target: :price, split: {column: :listed_at, value: Date.today - 20})
    end
    assert_equal "No data in training set", error.message
  end

  def test_split_no_validation_data
    data = houses_data.map { |r| r.slice(:bedrooms, :bathrooms, :price).merge(listed_at: Date.today - rand(10)) }
    model = Eps::Model.new
    error = assert_raises do
      model.train(data, target: :price, split: {column: :listed_at, value: Date.today + 20})
    end
    assert_equal "No data in validation set", error.message
  end
end
