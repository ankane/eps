require "bundler/setup"
Bundler.require(:default)
require "minitest/autorun"
require "minitest/pride"
require "daru"
require "gsl" if ENV["GSL"]

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
end
