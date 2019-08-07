require_relative "test_helper"

class DataFrameTest < Minitest::Test
  def test_subsetting
    df = Eps::DataFrame.new(c1: ["a", "b", "c"], c2: [1, 2, 3])
    assert_equal df, df[0..]
    assert_equal Eps::DataFrame.new(c1: ["b"], c2: [2]), df[1]
    assert_equal Eps::DataFrame.new(c2: [2, 3]), df[1.., "c2"]
  end
end
