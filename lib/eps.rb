# dependencies
require "lightgbm"
require "matrix"
require "nokogiri"

# stdlib
require "json"

# modules
require_relative "eps/base"
require_relative "eps/base_estimator"
require_relative "eps/data_frame"
require_relative "eps/label_encoder"
require_relative "eps/lightgbm"
require_relative "eps/linear_regression"
require_relative "eps/metrics"
require_relative "eps/model"
require_relative "eps/naive_bayes"
require_relative "eps/statistics"
require_relative "eps/text_encoder"
require_relative "eps/utils"
require_relative "eps/version"

# pmml
require_relative "eps/pmml"
require_relative "eps/pmml/generator"
require_relative "eps/pmml/loader"

# evaluators
require_relative "eps/evaluators/linear_regression"
require_relative "eps/evaluators/lightgbm"
require_relative "eps/evaluators/naive_bayes"
require_relative "eps/evaluators/node"

module Eps
  class Error < StandardError; end
  class UnstableSolution < Error; end

  def self.metrics(y_true, y_pred, weight: nil)
    if Utils.column_type(y_true, "actual") == "numeric"
      {
        rmse: Metrics.rmse(y_true, y_pred, weight: weight),
        mae: Metrics.mae(y_true, y_pred, weight: weight),
        me: Metrics.me(y_true, y_pred, weight: weight)
      }
    else
      {
        accuracy: Metrics.accuracy(y_true, y_pred, weight: weight)
      }
    end
  end

  # backwards compatibility
  Regressor = Model
end
