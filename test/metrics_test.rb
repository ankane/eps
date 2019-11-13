require_relative "test_helper"

class MetricsTest < Minitest::Test
  def test_regression_metrics
    actual = [1, 2, 3]
    estimated = [1, 2, 9]
    metrics = Eps.metrics(actual, estimated)

    assert_in_delta 3.464, metrics[:rmse]
    assert_in_delta 2, metrics[:mae]
    assert_in_delta -2, metrics[:me]
  end

  def test_regression_metrics_weight
    actual = [1, 1, 1]
    estimated = [0, 1, 4]
    weight = [1, 2, 3]
    metrics = Eps.metrics(actual, estimated, weight: weight)

    assert_in_delta 2.160246899469287, metrics[:rmse]
    assert_in_delta 1.6667, metrics[:mae]
    assert_in_delta -1.3333, metrics[:me]
  end

  def test_classification_metrics
    actual = ["up", "up", "down"]
    estimated = ["down", "up", "down"]
    metrics = Eps.metrics(actual, estimated)

    assert_in_delta 0.667, metrics[:accuracy]
  end

  def test_classification_metrics_weight
    actual = ["down", "up", "down"]
    estimated = ["down", "up", "up"]
    weight = [1, 2, 3]
    metrics = Eps.metrics(actual, estimated, weight: weight)

    assert_in_delta 0.5, metrics[:accuracy]
  end

  def test_log_loss_weight
    log_loss = Eps::Metrics.log_loss([0, 1, 0], [0, 1, 1], weight: [1, 2, 3])
    assert_in_delta 17.26978799617044, log_loss
  end
end
