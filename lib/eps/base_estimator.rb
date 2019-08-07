module Eps
  class BaseEstimator
    def train(data, y, target: nil, **options)
      x = normalize_x(data)
      y = y.to_a

      # check data
      raise "No data" if x.empty?
      raise "Number of samples differs from target" if x.size != y.size
      check_missing(y)

      @x = x
      @y = y
      @target = target || "target"

      # determine feature types
      @features = {}
      x.columns.each do |k, v|
        next if k == target
        @features[k] = v[0].is_a?(Numeric) ? "numeric" : "categorical"
      end
    end

    def predict(x)
      singular = x.is_a?(Hash)
      x = [x] if singular

      x = normalize_x(x)
      pred = _predict(x)

      singular ? pred[0] : pred
    end

    def evaluate(data, y = nil, target: nil)
      target ||= @target
      raise ArgumentError, "missing target" if !target && !y

      actual = y || Eps::DataFrame.new(data).columns[target.to_s]

      # check data
      # raise "No data" if data.empty?
      raise "Number of samples differs from target" if data.size != actual.size
      check_missing(actual)

      estimated = predict(data)

      self.class.metrics(actual, estimated)
    end

    private

    def check_missing(y)
      raise "Target missing in data" if y.any?(&:nil?)
    end

    def normalize_x(x)
      return x if x.is_a?(Eps::DataFrame)

      Eps::DataFrame.new(x)
    end
  end
end
