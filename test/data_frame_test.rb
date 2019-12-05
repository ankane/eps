require_relative "test_helper"

class DataFrameTest < Minitest::Test
  def test_subsetting
    df = Eps::DataFrame.new(c1: ["a", "b", "c"], c2: [1, 2, 3])
    assert_equal df, df[0..nil]
    assert_equal Eps::DataFrame.new(c1: ["b"], c2: [2]), df[1]
    assert_equal Eps::DataFrame.new(c2: [2, 3]), df[1..nil, ["c2"]]
    assert_equal Eps::DataFrame.new(c1: ["c"], c2: [3]), df[-1]
    assert_equal [2, 3], df[1..nil, "c2"]
    assert_equal df, df[0..nil, "c1".."c2"]
    refute_equal df, df[0..nil, "c2".."c1"]
    assert_equal df[["c1"]], df[0..nil, "c1"..."c2"]
    assert_equal Eps::DataFrame.new(c2: [3], c1: ["c"]), df[2..nil, "c2".."c1"]
    assert_equal ["a", "b", "c"], df["c1"]
    assert_equal Eps::DataFrame.new(c1: ["a", "b", "c"]), df[["c1"]]
    assert_equal df[0..1], df[0...-1]
    assert_equal Eps::DataFrame.new(c1: ["a", "c"], c2: [1, 3]), df[[0, 2]]
    assert_equal df, df[0...10]

    error = assert_raises do
      df[0..nil, "c3"]
    end
    assert_equal "Undefined column: c3", error.message

    error = assert_raises do
      df[0..nil, "c1".."c3"]
    end
    assert_equal "Undefined column: c3", error.message

    error = assert_raises do
      df[0..nil, "c3".."c2"]
    end
    assert_equal "Undefined column: c3", error.message
  end
end
