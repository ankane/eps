require "bundler/setup"
Bundler.require(:default)
require "minitest/autorun"
require "minitest/pride"
require "json"
require "nokogiri"
require "daru"
require "gsl" if ENV["GSL"]
