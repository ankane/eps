# dependencies
require "json"
require "matrix"
require "nokogiri"

# modules
require "eps/base"
require "eps/base_estimator"
require "eps/data_frame"
require "eps/evaluators/linear_regression"
require "eps/evaluators/naive_bayes"
require "eps/linear_regression"
require "eps/metrics"
require "eps/model"
require "eps/naive_bayes"
require "eps/onnx"
require "eps/statistics"
require "eps/text_encoder"
require "eps/utils"
require "eps/version"

module Eps
  def self.metrics(y_true, y_pred)
    if Utils.column_type(y_true, "y_true") == "numeric"
      {
        rmse: Metrics.rmse(y_true, y_pred),
        mae: Metrics.mae(y_true, y_pred),
        me: Metrics.me(y_true, y_pred)
      }
    else
      {
        accuracy: Metrics.accuracy(y_true, y_pred)
      }
    end
  end

  # backwards compatibility
  Regressor = Model
end
