require "bundler/setup"
Bundler.require(:default)
require "minitest/autorun"
require "minitest/pride"
require "daru"
require "numo/narray"
require "rover"

class Minitest::Test
  def assert_valid_pmml(pmml)
    xsd = Nokogiri::XML::Schema(File.read("test/support/pmml-4-4.xsd"))
    doc = Nokogiri::XML(pmml)
    assert_empty xsd.validate(doc)
  end

  def assert_elements_in_delta(expected, actual)
    assert_equal expected.size, actual.size
    expected.zip(actual) do |exp, act|
      assert_in_delta exp, act
    end
  end

  def mpg_data(binary: false, extra_fields: [])
    data = CSV.table("test/support/mpg.csv")
    fields = [:drv, :class, :displ, :year, :cyl, :hwy] + extra_fields
    data = data.map { |row| row.to_h.slice(*fields) }
    data.each do |d|
      d[:drv] = "4" if d[:drv] == "r" if binary
      d[:drv] = d[:drv].to_s
    end
    data
  end
end
