module Eps
  class LinearRegression < BaseEstimator
    def initialize(coefficients: nil, gsl: nil)
      @coefficients = Hash[coefficients.map { |k, v| [k.is_a?(Array) ? [k[0].to_sym, k[1]] : k.to_sym, v] }] if coefficients
      @gsl = gsl.nil? ? defined?(GSL) : gsl
    end

    def train(*args)
      super

      x, @coefficient_names = prep_x(@x)

      if x.size <= @coefficient_names.size
        raise "Number of samples must be at least two more than number of features"
      end

      v3 =
        if @gsl
          x = GSL::Matrix.alloc(*x)
          y = GSL::Vector.alloc(@y)
          c, @covariance, _, _ = GSL::MultiFit::linear(x, y)
          c.to_a
        else
          x = Matrix.rows(x)
          y = Matrix.column_vector(@y)
          removed = []

          # https://statsmaths.github.io/stat612/lectures/lec13/lecture13.pdf
          # unforutnately, this method is unstable
          # haven't found an efficient way to do QR-factorization in Ruby
          # the extendmatrix gem has householder and givens (givens has bug)
          # but methods are too slow
          xt = x.t
          begin
            @xtxi = (xt * x).inverse
          rescue ExceptionForMatrix::ErrNotRegular
            constant = {}
            (1...x.column_count).each do |i|
              constant[i] = constant?(x.column(i))
            end

            # remove constant columns
            removed = constant.select { |_, v| v }.keys

            # remove non-independent columns
            constant.select { |_, v| !v }.keys.combination(2) do |c2|
              if !x.column(c2[0]).independent?(x.column(c2[1]))
                removed << c2[1]
              end
            end

            vectors = x.column_vectors
            # delete in reverse of indexes stay the same
            removed.sort.reverse.each do |i|
              # @coefficient_names.delete_at(i)
              vectors.delete_at(i)
            end
            x = Matrix.columns(vectors)
            xt = x.t

            # try again
            begin
              @xtxi = (xt * x).inverse
            rescue ExceptionForMatrix::ErrNotRegular
              raise "Multiple solutions - GSL is needed to select one"
            end
          end
          # huge performance boost
          # by multiplying xt * y first
          v2 = matrix_arr(@xtxi * (xt * y))

          # add back removed
          removed.sort.each do |i|
            v2.insert(i, 0)
          end
          @removed = removed

          v2
        end

      @coefficients = Hash[@coefficient_names.zip(v3)]
    end

    # legacy

    def coefficients
      Hash[@coefficients.map { |k, v| [Array(k).join.to_sym, v] }]
    end

    # ruby

    def self.load(data)
      new(Hash[data.map { |k, v| [k.to_sym, v] }])
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

      new(coefficients: coefficients)
    end

    def to_json
      JSON.generate(dump)
    end

    # pmml

    def self.load_pmml(data)
      # TODO more validation
      node = data.css("RegressionTable")
      coefficients = {
        _intercept: node.attribute("intercept").value.to_f
      }
      node.css("NumericPredictor").each do |n|
        coefficients[n.attribute("name").value] = n.attribute("coefficient").value.to_f
      end
      node.css("CategoricalPredictor").each do |n|
        coefficients[[n.attribute("name").value.to_sym, n.attribute("value").value]] = n.attribute("coefficient").value.to_f
      end
      new(coefficients: coefficients)
    end

    def to_pmml
      predictors = @coefficients.reject { |k| k == :_intercept }

      data_fields = {}
      predictors.each do |k, v|
        if k.is_a?(Array)
          (data_fields[k[0]] ||= []) << k[1]
        else
          data_fields[k] = nil
        end
      end

      builder = Nokogiri::XML::Builder.new do |xml|
        xml.PMML(version: "4.3", xmlns: "http://www.dmg.org/PMML-4_3", "xmlns:xsi" => "http://www.w3.org/2001/XMLSchema-instance") do
          xml.Header
          xml.DataDictionary do
            data_fields.each do |k, vs|
              if vs
                xml.DataField(name: k, optype: "categorical", dataType: "string") do
                  vs.each do |v|
                    xml.Value(value: v)
                  end
                end
              else
                xml.DataField(name: k, optype: "continuous", dataType: "double")
              end
            end
          end
          xml.RegressionModel(functionName: "regression") do
            xml.MiningSchema do
              data_fields.each do |k, _|
                xml.MiningField(name: k)
              end
            end
            xml.RegressionTable(intercept: @coefficients[:_intercept]) do
              predictors.each do |k, v|
                if k.is_a?(Array)
                  xml.CategoricalPredictor(name: k[0], value: k[1], coefficient: v)
                else
                  xml.NumericPredictor(name: k, coefficient: v)
                end
              end
            end
          end
        end
      end.to_xml
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
      new(coefficients: coefficients)
    end

    # metrics

    def self.metrics(actual, estimated)
      errors = actual.zip(estimated).map { |yi, yi2| yi - yi2 }

      {
        me: mean(errors),
        mae: mean(errors.map { |v| v.abs }),
        rmse: Math.sqrt(mean(errors.map { |v| v**2 }))
      }
    end

    # private
    def self.mean(arr)
      arr.inject(0, &:+) / arr.size.to_f
    end

    # https://people.richland.edu/james/ictcm/2004/multiple.html
    def summary(extended: false)
      @summary_str ||= begin
        str = String.new("")
        len = [coefficients.keys.map(&:size).max, 15].max
        if extended
          str += "%-#{len}s %12s %12s %12s %12s\n" % ["", "coef", "stderr", "t", "p"]
        else
          str += "%-#{len}s %12s %12s\n" % ["", "coef", "p"]
        end
        coefficients.each do |k, v|
          if extended
            str += "%-#{len}s %12.2f %12.2f %12.2f %12.3f\n" % [display_field(k), v, std_err[k], t_value[k], p_value[k]]
          else
            str += "%-#{len}s %12.2f %12.3f\n" % [display_field(k), v, p_value[k]]
          end
        end
        str += "\n"
        str += "r2: %.3f\n" % [r2] if extended
        str += "adjusted r2: %.3f\n" % [adjusted_r2]
        str
      end
    end

    def r2
      @r2 ||= (sst - sse) / sst
    end

    def adjusted_r2
      @adjusted_r2 ||= (mst - mse) / mst
    end

    private

    def _predict(x)
      x, c = prep_x(x, train: false)
      coef = c.map do |v|
        # use 0 if coefficient does not exist
        # this can happen for categorical features
        # since only n-1 coefficients are stored
        @coefficients[v] || 0
      end

      x = Matrix.rows(x)
      c = Matrix.column_vector(coef)
      matrix_arr(x * c)
    end

    def display_field(k)
      k.is_a?(Array) ? k.join("") : k
    end

    def constant?(arr)
      arr.all? { |x| x == arr[0] }
    end

    # add epsilon for perfect fits
    # consistent with GSL
    def t_value
      @t_value ||= Hash[coefficients.map { |k, v| [k, v / (std_err[k] + Float::EPSILON)] }]
    end

    def p_value
      @p_value ||= begin
        Hash[coefficients.map do |k, _|
          tp =
            if @gsl
              GSL::Cdf.tdist_P(t_value[k].abs, degrees_of_freedom)
            else
              tdist_p(t_value[k].abs, degrees_of_freedom)
            end

          [k, 2 * (1 - tp)]
        end]
      end
    end

    def std_err
      @std_err ||= begin
        Hash[@coefficient_names.zip(diagonal.map { |v| Math.sqrt(v) })]
      end
    end

    def diagonal
      @diagonal ||= begin
        if covariance.respond_to?(:each)
          d = covariance.each(:diagonal).to_a
          @removed.each do |i|
            d.insert(i, 0)
          end
          d
        else
          covariance.diagonal.to_a
        end
      end
    end

    def covariance
      @covariance ||= mse * @xtxi
    end

    def y_bar
      @y_bar ||= mean(@y)
    end

    def y_hat
      @y_hat ||= predict(@x)
    end

    # total sum of squares
    def sst
      @sst ||= @y.map { |y| (y - y_bar)**2 }.sum
    end

    # sum of squared errors of prediction
    # not to be confused with "explained sum of squares"
    def sse
      @sse ||= @y.zip(y_hat).map { |y, yh| (y - yh)**2 }.sum
    end

    def mst
      @mst ||= sst / (@y.size - 1)
    end

    def mse
      @mse ||= sse / degrees_of_freedom
    end

    def degrees_of_freedom
      @y.size - coefficients.size
    end

    def mean(arr)
      arr.sum / arr.size.to_f
    end

    ### Extracted from https://github.com/estebanz01/ruby-statistics
    ### The Ruby author is Esteban Zapata Rojas
    ###
    ### Originally extracted from https://codeplea.com/incomplete-beta-function-c
    ### This function is shared under zlib license and the author is Lewis Van Winkle
    def tdist_p(value, degrees_of_freedom)
      upper = (value + Math.sqrt(value * value + degrees_of_freedom))
      lower = (2.0 * Math.sqrt(value * value + degrees_of_freedom))

      x = upper/lower

      alpha = degrees_of_freedom/2.0
      beta = degrees_of_freedom/2.0

      incomplete_beta_function(x, alpha, beta)
    end

    ### Extracted from https://github.com/estebanz01/ruby-statistics
    ### The Ruby author is Esteban Zapata Rojas
    ###
    ### This implementation is an adaptation of the incomplete beta function made in C by
    ### Lewis Van Winkle, which released the code under the zlib license.
    ### The whole math behind this code is described in the following post: https://codeplea.com/incomplete-beta-function-c
    def incomplete_beta_function(x, alp, bet)
      return if x < 0.0
      return 1.0 if x > 1.0

      tiny = 1.0E-50

      if x > ((alp + 1.0)/(alp + bet + 2.0))
        return 1.0 - incomplete_beta_function(1.0 - x, bet, alp)
      end

      # To avoid overflow problems, the implementation applies the logarithm properties
      # to calculate in a faster and safer way the values.
      lbet_ab = (Math.lgamma(alp)[0] + Math.lgamma(bet)[0] - Math.lgamma(alp + bet)[0]).freeze
      front = (Math.exp(Math.log(x) * alp + Math.log(1.0 - x) * bet - lbet_ab) / alp.to_f).freeze

      # This is the non-log version of the left part of the formula (before the continuous fraction)
      # down_left = alp * self.beta_function(alp, bet)
      # upper_left = (x ** alp) * ((1.0 - x) ** bet)
      # front = upper_left/down_left

      f, c, d = 1.0, 1.0, 0.0

      returned_value = nil

      # Let's do more iterations than the proposed implementation (200 iters)
      (0..500).each do |number|
        m = number/2

        numerator = if number == 0
                      1.0
                    elsif number % 2 == 0
                      (m * (bet - m) * x)/((alp + 2.0 * m - 1.0)* (alp + 2.0 * m))
                    else
                      top = -((alp + m) * (alp + bet + m) * x)
                      down = ((alp + 2.0 * m) * (alp + 2.0 * m + 1.0))

                      top/down
                    end

        d = 1.0 + numerator * d
        d = tiny if d.abs < tiny
        d = 1.0 / d

        c = 1.0 + numerator / c
        c = tiny if c.abs < tiny

        cd = (c*d).freeze
        f = f * cd

        if (1.0 - cd).abs < 1.0E-10
          returned_value = front * (f - 1.0)
          break
        end
      end

      returned_value
    end

    def prep_x(x, train: true)
      coefficients = @coefficients

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

      # get column types
      if train
        column_types = {}
        if x.any?
          row = x.first
          row.each do |k, v|
            column_types[k] = categorical?(v) ? "categorical" : "numeric"
          end
        end
      else
        # get column types for prediction
        column_types = {}
        coefficients.each do |k, v|
          next if k == :_intercept
          if k.is_a?(Array)
            column_types[k.first] = "categorical"
          else
            column_types[k] = "numeric"
          end
        end
      end

      # if !train && x.any?
      #   # check first row against coefficients
      #   ckeys = coefficients.keys.map(&:to_s)
      #   bad_keys = x[0].keys.map(&:to_s).reject { |k| ckeys.any? { |c| c.start_with?(k) } }
      #   raise "Unknown keys: #{bad_keys.join(", ")}" if bad_keys.any?
      # end

      supports_categorical = train || coefficients.any? { |k, _| k.is_a?(Array) }

      cache = {}
      first_key = {}
      i = 0
      rows = []
      x.each do |xi|
        row = {}
        xi.each do |k, v|
          categorical = column_types[k.to_sym] == "categorical" || (!supports_categorical && categorical?(v))

          key = categorical ? [k.to_sym, v] : k.to_sym
          v2 = categorical ? 1 : v

          # TODO make more efficient
          check_key = supports_categorical ? key : symbolize_coef(key)
          next if !train && !coefficients.key?(check_key)

          raise "Missing data" if v2.nil?

          unless cache[key]
            cache[key] = i
            first_key[k] ||= key if categorical
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
      c = [:_intercept] + cache.sort_by { |_, v| v }.map(&:first)

      unless supports_categorical
        c = c.map { |v| symbolize_coef(v) }
      end

      [ret2, c]
    end

    def symbolize_coef(k)
      (k.is_a?(Array) ? k.join("") : k).to_sym
    end

    def matrix_arr(matrix)
      matrix.to_a.map { |xi| xi[0].to_f }
    end
  end
end
