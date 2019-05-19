module Eps
  class BaseEstimator
    def train(data, y, target: nil, **options)
      # TODO more performant conversion
      if daru?(data)
        x = data.dup
        x = x.delete_vector(target) if target
      else
        x = data.map(&:dup)
        x.each { |r| r.delete(target) } if target
      end

      y = prep_y(y.to_a)

      @target = target || "target"

      if x.size != y.size
        raise "Number of samples differs from target"
      end

      @x = x
      @y = y
      @target = target
    end

    def predict(x)
      singular = !(x.is_a?(Array) || daru?(x))
      x = [x] if singular

      pred = _predict(x)

      singular ? pred[0] : pred
    end

    def evaluate(data, y = nil, target: nil)
      target ||= @target
      raise ArgumentError, "missing target" if !target && !y

      actual = y
      actual ||=
        if daru?(data)
          data[target].to_a
        else
          data.map { |v| v[target] }
        end

      actual = prep_y(actual)
      estimated = predict(data)

      self.class.metrics(actual, estimated)
    end

    private

    def categorical?(v)
      !v.is_a?(Numeric)
    end

    def daru?(x)
      defined?(Daru) && x.is_a?(Daru::DataFrame)
    end

    def flip_target(target)
      target.is_a?(String) ? target.to_sym : target.to_s
    end

    def prep_y(y)
      y.each do |yi|
        raise "Target missing in data" if yi.nil?
      end
      y
    end

    # determine if target is a string or symbol
    def prep_target(target, data)
      if daru?(data)
        data.has_vector?(target) ? target : flip_target(target)
      else
        x = data[0] || {}
        x[target] ? target : flip_target(target)
      end
    end
  end
end
