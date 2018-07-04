module Eps
  class Regressor < BaseRegressor
    def initialize(data, y = nil, target: nil, gsl: nil)
      raise ArgumentError, "missing target" if !target && !y

      target = prep_target(target, data) if target

      # TODO more performant conversion
      if daru?(data)
        y ||= data[target].to_a
        x = data.dup.delete_vector(target)
      else
        x = data.map(&:dup)
        y ||= x.map { |v| v.delete(target) }
      end

      @x = x
      @y = prep_y(y.to_a)

      if @x.size != @y.size
        raise "Number of samples differs from target"
      end

      @target = target
      @gsl = gsl.nil? ? defined?(GSL) : gsl

      # fit immediately
      coefficients
    end

    def coefficients
      @coefficients ||= begin
        x, @coefficient_names = prep_x(@x)

        if x.size <= @coefficient_names.size
          raise "Number of samples must be at least two more than number of features"
        end

        v =
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
              constant.select { |_, v| !v }.keys.combination(2) do |c|
                if !x.column(c[0]).independent?(x.column(c[1]))
                  removed << c[1]
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

        Hash[@coefficient_names.zip(v)]
      end
    end

    def evaluate(data, y = nil)
      super(data, y, target: @target)
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
            str += "%-#{len}s %12.2f %12.2f %12.2f %12.3f\n" % [k, v, std_err[k], t_value[k], p_value[k]]
          else
            str += "%-#{len}s %12.2f %12.3f\n" % [k, v, p_value[k]]
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
      @sst ||= sum(@y.map { |y| (y - y_bar)**2 })
    end

    # sum of squared errors of prediction
    # not to be confused with "explained sum of squares"
    def sse
      @sse ||= sum(@y.zip(y_hat).map { |y, yh| (y - yh)**2 })
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

    def sum(arr)
      arr.inject(0, &:+)
    end

    def mean(arr)
      sum(arr) / arr.size.to_f
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
  end
end
