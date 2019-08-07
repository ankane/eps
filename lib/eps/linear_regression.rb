module Eps
  class LinearRegression < BaseEstimator
    def initialize(coefficients: nil, features: nil, gsl: nil)
      @coefficients = Hash[coefficients.map { |k, v| [k.is_a?(Array) ? [k[0].to_sym, k[1]] : k.to_sym, v] }] if coefficients
      @features = features

      # legacy
      if @coefficients && !@features
        @features = Hash[@coefficients.keys.map { |k| [k.to_s, "numeric"] }]
        @features.delete("_intercept")
      end

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

    def to_json(opts = {})
      JSON.generate(dump, opts)
    end

    # pmml

    def self.load_pmml(data)
      # TODO more validation
      node = data.css("RegressionTable")
      coefficients = {
        _intercept: node.attribute("intercept").value.to_f
      }
      features = {}
      node.css("NumericPredictor").each do |n|
        name = n.attribute("name").value
        coefficients[name] = n.attribute("coefficient").value.to_f
        features[name] = "numeric"
      end
      node.css("CategoricalPredictor").each do |n|
        name = n.attribute("name").value
        coefficients[[name, n.attribute("value").value]] = n.attribute("coefficient").value.to_f
        features[name] = "categorical"
      end
      new(coefficients: coefficients, features: features)
    end

    def to_pmml
      predictors = @coefficients.reject { |k| k == :_intercept }

      data_fields = {}
      predictors.keys.each do |k|
        if k.is_a?(Array)
          (data_fields[k[0]] ||= []) << k[1]
        else
          data_fields[k] = nil
        end
      end

      Nokogiri::XML::Builder.new do |xml|
        xml.PMML(version: "4.4", xmlns: "http://www.dmg.org/PMML-4_4", "xmlns:xsi" => "http://www.w3.org/2001/XMLSchema-instance") do
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
      coefficients = @coefficients
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
      @t_value ||= Hash[@coefficients.map { |k, v| [k, v / (std_err[k] + Float::EPSILON)] }]
    end

    def p_value
      @p_value ||= begin
        Hash[@coefficients.map do |k, _|
          tp =
            if @gsl
              GSL::Cdf.tdist_P(t_value[k].abs, degrees_of_freedom)
            else
              Eps::Statistics.tdist_p(t_value[k].abs, degrees_of_freedom)
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
      @y.size - @coefficients.size
    end

    def mean(arr)
      arr.sum / arr.size.to_f
    end

    def prep_x(x, train: true)
      matrix = []
      column_names = []

      # intercept
      x.size.times do
        matrix << [1]
      end
      column_names << :_intercept

      @features.each do |k, type|
        # if type == "numeric" && !train && !@features.any? { |k, v| v == "categorical" } && !x.first[k].is_a?(Numeric)
        #   # legacy could be categorical
        #   type = "categorical"
        # end

        raise "Missing data in #{k}" if !x.columns[k] || x.columns[k].any?(&:nil?)

        if type == "numeric"
          x.columns[k].zip(matrix) do |xi, mi|
            mi << xi
          end
          column_names << k.to_sym
        else
          if train
            values = x.columns[k].uniq
            # n - 1 dummy variables
            values.shift if train
          else
            # get from coefficients
            values = @coefficients.select { |k2, _| k2.is_a?(Array) && k2[0].to_s == k }.map { |k2, _| k2[1] }
          end

          # get index to set
          indexes = {}
          offset = column_names.size
          values.each do |v|
            indexes[v] = offset
            offset += 1
          end

          zeros = [0] * values.size
          x.columns[k].zip(matrix) do |xi, mi|
            mi.concat(zeros)
            off = indexes[xi]
            mi[off] = 1 if off
          end
          column_names.concat(values.map { |v| [k.to_sym, v] })
        end
      end

      [matrix, column_names]
    end

    def matrix_arr(matrix)
      matrix.to_a.map { |xi| xi[0].to_f }
    end
  end
end
