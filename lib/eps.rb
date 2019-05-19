# dependencies
require "matrix"
require "json"

# modules
require "eps/base"
require "eps/base_estimator"
require "eps/linear_regression"
require "eps/model"
require "eps/naive_bayes"
require "eps/version"

module Eps
  def self.metrics(actual, estimated)
    Eps::Model.metrics(actual, estimated)
  end

  # backwards compatibility
  Regressor = Model
end
