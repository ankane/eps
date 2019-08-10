require "bundler/setup"
Bundler.require(:default)
require "minitest/autorun"
require "minitest/pride"
require "daru"
require "nokogiri"
require "gsl" if ENV["GSL"]

class Minitest::Test
  def assert_valid_pmml(pmml)
    xsd = Nokogiri::XML::Schema(File.read("test/support/pmml-4-4.xsd"))
    doc = Nokogiri::XML(pmml)
    assert_empty xsd.validate(doc)
  end
end
