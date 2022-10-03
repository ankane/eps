module Eps
  class LinearRegression < BaseEstimator
    def coefficients
      @evaluator.coefficients
    end

    def r2
      @r2 ||= (sst - sse) / sst
    end

    def adjusted_r2
      @adjusted_r2 ||= (mst - mse) / mst
    end

    private

    # https://people.richland.edu/james/ictcm/2004/multiple.html
    def _summary(extended: false)
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

    def _train(intercept: true, gsl: nil)
      raise "Target must be numeric" if @target_type != "numeric"
      check_missing_value(@train_set)
      check_missing_value(@validation_set) if @validation_set

      data = prep_x(@train_set)

      if data.size < data.columns.size + 2
        raise "Number of data points must be at least two more than number of features"
      end

      x = data.map_rows(&:to_a)

      gsl = defined?(GSLR) if gsl.nil?

      if intercept && !gsl
        data.size.times do |i|
          x[i].unshift(1)
        end
      end

      v3 =
        if gsl
          model = GSLR::OLS.new(intercept: intercept)
          model.fit(x, data.label, weight: data.weight)

          @covariance = model.covariance

          coefficients = model.coefficients.dup
          coefficients.unshift(model.intercept) if intercept
          coefficients
        else
          x = Matrix.rows(x)
          y = Matrix.column_vector(data.label)

          # weighted OLS
          # http://www.real-statistics.com/multiple-regression/weighted-linear-regression/weighted-regression-basics/
          w = Matrix.diagonal(*data.weight) if data.weight

          removed = []

          # https://statsmaths.github.io/stat612/lectures/lec13/lecture13.pdf
          # unfortunately, this method is unstable
          # haven't found an efficient way to do QR-factorization in Ruby
          # the extendmatrix gem has householder and givens (givens has bug)
          # but methods are too slow
          xt = x.t
          xt *= w if w
          begin
            @xtxi = (xt * x).inverse
          rescue ExceptionForMatrix::ErrNotRegular
            # matrix cannot be inverted
            # https://en.wikipedia.org/wiki/Multicollinearity

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
            xt *= w if w

            # try again
            begin
              @xtxi = (xt * x).inverse
            rescue ExceptionForMatrix::ErrNotRegular
              raise "Multiple solutions - GSL is needed to select one"
            end
          end
          # huge performance boost
          # by multiplying xt * y first
          # for weighted, w is already included in wt
          v2 = @xtxi * (xt * y)

          # convert to array
          v2 = v2.to_a.map { |xi| xi[0].to_f }

          # add back removed
          removed.sort.each do |i|
            v2.insert(i, 0)
          end
          @removed = removed

          v2
        end

      if @xtxi && @xtxi.each(:diagonal).any? { |v| v < 0 }
        raise UnstableSolution, "GSL is needed to find a stable solution for this dataset"
      end

      @coefficient_names = data.columns.keys
      @coefficient_names.unshift("_intercept") if intercept
      @coefficients = @coefficient_names.zip(v3).to_h
      Evaluators::LinearRegression.new(coefficients: @coefficients, features: @features, text_features: @text_features)
    end

    def prep_x(x)
      x = x.dup
      @features.each do |k, type|
        if type == "categorical"
          values = x.columns.delete(k)
          labels = values.uniq[1..-1]
          labels.each do |label|
            x.columns[[k, label]] = values.map { |v| v == label ? 1 : 0 }
          end
        end
      end
      prep_text_features(x)
      x
    end

    def constant?(arr)
      arr.all? { |x| x == arr[0] }
    end

    # add epsilon for perfect fits
    # consistent with GSL
    def t_value
      @t_value ||= @coefficients.to_h { |k, v| [k, v / (std_err[k] + Float::EPSILON)] }
    end

    def p_value
      @p_value ||= begin
        @coefficients.to_h do |k, _|
          tp = Eps::Statistics.tdist_p(t_value[k].abs, degrees_of_freedom)
          [k, 2 * (1 - tp)]
        end
      end
    end

    def std_err
      @std_err ||= begin
        @coefficient_names.zip(diagonal.map { |v| Math.sqrt(v) }).to_h
      end
    end

    def diagonal
      @diagonal ||= begin
        if covariance.is_a?(Array)
          covariance.size.times.map do |i|
            covariance[i][i]
          end
        elsif covariance.respond_to?(:each)
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
      @y_bar ||= mean(@train_set.label)
    end

    def y_hat
      @y_hat ||= predict(@train_set)
    end

    # total sum of squares
    def sst
      @sst ||= @train_set.label.map { |y| (y - y_bar)**2 }.sum
    end

    # sum of squared errors of prediction
    # not to be confused with "explained sum of squares"
    def sse
      @sse ||= @train_set.label.zip(y_hat).map { |y, yh| (y - yh)**2 }.sum
    end

    def mst
      @mst ||= sst / (@train_set.size - 1)
    end

    def mse
      @mse ||= sse / degrees_of_freedom
    end

    def degrees_of_freedom
      @train_set.size - @coefficients.size
    end

    def mean(arr)
      arr.sum / arr.size.to_f
    end
  end
end
