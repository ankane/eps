### Extracted from https://github.com/estebanz01/ruby-statistics
### The Ruby author is Esteban Zapata Rojas
###
### Originally extracted from https://codeplea.com/incomplete-beta-function-c
### These functions shared under zlib license and the author is Lewis Van Winkle

module Eps
  module Statistics
    def self.tdist_p(value, degrees_of_freedom)
      upper = (value + Math.sqrt(value * value + degrees_of_freedom))
      lower = (2.0 * Math.sqrt(value * value + degrees_of_freedom))

      x = upper/lower

      alpha = degrees_of_freedom/2.0
      beta = degrees_of_freedom/2.0

      incomplete_beta_function(x, alpha, beta)
    end

    def self.incomplete_beta_function(x, alp, bet)
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
