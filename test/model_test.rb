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
end
