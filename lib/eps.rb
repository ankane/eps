# dependencies
require "json"
require "lightgbm"
require "matrix"
require "nokogiri"

# modules
require "eps/base"
require "eps/base_estimator"
require "eps/data_frame"
require "eps/label_encoder"
require "eps/lightgbm"
require "eps/linear_regression"
require "eps/metrics"
require "eps/model"
require "eps/naive_bayes"
require "eps/statistics"
require "eps/text_encoder"
require "eps/utils"
require "eps/version"

# pmml
require "eps/pmml"
require "eps/pmml/generator"
require "eps/pmml/loader"

# evaluators
require "eps/evaluators/linear_regression"
require "eps/evaluators/lightgbm"
require "eps/evaluators/naive_bayes"
require "eps/evaluators/node"

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
