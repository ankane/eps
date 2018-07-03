# dependencies
require "matrix"
require "json"

# modules
require "eps/base_regressor"
require "eps/metrics"
require "eps/regressor"
require "eps/version"

module Eps
  def self.metrics(actual, estimated)
    Eps::Metrics.new(actual, estimated).all
  end
end
