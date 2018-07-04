module Eps
  class BaseRegressor
    attr_reader :coefficients

    def initialize(coefficients:)
      @coefficients = Hash[coefficients.map { |k, v| [k.to_sym, v] }]
    end

    def predict(x)
      singular = !(x.is_a?(Array) || daru?(x))
      x = [x] if singular
      x, c = prep_x(x, train: false)
      coef = c.map do |v|
        # use 0 if coefficient does not exist
        # this can happen for categorical features
        # since only n-1 coefficients are stored
        coefficients[v] || 0
      end

      x = Matrix.rows(x)
      c = Matrix.column_vector(coef)
      pred = matrix_arr(x * c)

      singular ? pred[0] : pred
    end

    def evaluate(data, y = nil, target: nil)
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
      Eps.metrics(actual, estimated)
    end

    # ruby

    def self.load(data)
      BaseRegressor.new(Hash[data.map { |k, v| [k.to_sym, v] }])
    end

    def dump
      {coefficients: coefficients}
    end

    # json

    def self.load_json(data)
      data = JSON.parse(data) if data.is_a?(String)
      coefficients = data["coefficients"]

      # for R models
      if coefficients["(Intercept)"]
        coefficients = coefficients.dup
        coefficients["_intercept"] = coefficients.delete("(Intercept)")
      end

      BaseRegressor.new(coefficients: coefficients)
    end

    def to_json
      JSON.generate(dump)
    end

    # pmml

    def self.load_pmml(data)
      if data.is_a?(String)
        require "nokogiri"
        data = Nokogiri::XML(data)
      end

      # TODO more validation
      node = data.css("RegressionTable")
      coefficients = {
        _intercept: node.attribute("intercept").value.to_f
      }
      node.css("NumericPredictor").each do |n|
        coefficients[n.attribute("name").value] = n.attribute("coefficient").value.to_f
      end
      node.css("CategoricalPredictor").each do |n|
        coefficients["#{n.attribute("name").value}#{n.attribute("value").value}"] = n.attribute("coefficient").value.to_f
      end
      BaseRegressor.new(coefficients: coefficients)
    end

    # pfa

    def self.load_pfa(data)
      data = JSON.parse(data) if data.is_a?(String)
      init = data["cells"].first[1]["init"]
      names =
        if data["input"]["fields"]
          data["input"]["fields"].map { |f| f["name"] }
        else
          init["coeff"].map.with_index { |_, i| "x#{i}" }
        end
      coefficients = {
        _intercept: init["const"]
      }
      init["coeff"].each_with_index do |c, i|
        name = names[i]
        # R can export coefficients with same name
        raise "Coefficients with same name" if coefficients[name]
        coefficients[name] = c
      end
      BaseRegressor.new(coefficients: coefficients)
    end

    protected

    def daru?(x)
      defined?(Daru) && x.is_a?(Daru::DataFrame)
    end

    def prep_x(x, train: true)
      if daru?(x)
        x = x.to_a[0]
      else
        x = x.map do |xi|
          case xi
          when Hash
            xi
          when Array
            Hash[xi.map.with_index { |v, i| [:"x#{i}", v] }]
          else
            {x0: xi}
          end
        end
      end

      # if !train && x.any?
      #   # check first row against coefficients
      #   ckeys = coefficients.keys.map(&:to_s)
      #   bad_keys = x[0].keys.map(&:to_s).reject { |k| ckeys.any? { |c| c.start_with?(k) } }
      #   raise "Unknown keys: #{bad_keys.join(", ")}" if bad_keys.any?
      # end

      cache = {}
      first_key = {}
      i = 0
      rows = []
      x.each do |xi|
        row = {}
        xi.each do |k, v|
          key = v.is_a?(String) ? [k.to_sym, v] : k.to_sym
          v2 = v.is_a?(String) ? 1 : v

          # TODO make more efficient
          next if !train && !coefficients.key?(symbolize_coef(key))

          raise "Missing data" if v2.nil?

          unless cache[key]
            cache[key] = i
            first_key[k] ||= key if v.is_a?(String)
            i += 1
          end

          row[key] = v2
        end
        rows << row
      end

      if train
        # remove one degree of freedom
        first_key.values.each do |v|
          num = cache.delete(v)
          cache.each do |k, v2|
            cache[k] -= 1 if v2 > num
          end
        end
      end

      ret2 = []
      rows.each do |row|
        ret = [0] * cache.size
        row.each do |k, v|
          if cache[k]
            ret[cache[k]] = v
          end
        end
        ret2 << ([1] + ret)
      end

      # flatten keys
      c = [:_intercept] + cache.sort_by { |_, v| v }.map { |k, _| symbolize_coef(k) }

      if c.size != c.uniq.size
        raise "Overlapping coefficients"
      end

      [ret2, c]
    end

    def symbolize_coef(k)
      (k.is_a?(Array) ? k.join("") : k).to_sym
    end

    def matrix_arr(matrix)
      matrix.to_a.map { |xi| xi[0].to_f }
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

    def flip_target(target)
      target.is_a?(String) ? target.to_sym : target.to_s
    end

    def prep_y(y)
      y.each do |yi|
        raise "Target missing in data" if yi.nil?
      end
      y.map(&:to_f)
    end
  end
end
