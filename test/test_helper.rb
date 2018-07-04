require "bundler/setup"
Bundler.require(:default)
require "minitest/autorun"
require "minitest/pride"
require "daru"
require "gsl" if ENV["GSL"]
