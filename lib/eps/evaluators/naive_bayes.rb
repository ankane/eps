module Eps
  module Evaluators
    class NaiveBayes
      attr_reader :features, :probabilities

      def initialize(probabilities:, features:, derived: nil, legacy: false)
        @probabilities = probabilities
        @features = features
        @derived = derived
        @legacy = legacy
      end

      def predict(x, probabilities: false)
        probs = calculate_class_probabilities(x)
        probs.map do |xp|
          if probabilities
            sum = xp.values.map { |v| Math.exp(v) }.sum.to_f
            xp.map { |k, v| [k, Math.exp(v) / sum] }.to_h
          else
            xp.sort_by { |k, v| [-v, k] }[0][0]
          end
        end
      end

      # use log to prevent underflow
      # https://www.antoniomallia.it/lets-implement-a-gaussian-naive-bayes-classifier-in-python.html
      def calculate_class_probabilities(x)
        probs = Eps::DataFrame.new

        # assign very small probability if probability is 0
        tiny_p = @legacy ? 0.0001 : 0

        total = probabilities[:prior].values.sum.to_f
        probabilities[:prior].each do |c, cv|
          prior = Math.log(cv / total)
          px = [prior] * x.size

          @features.each do |k, type|
            case type
            when "categorical"
              x.columns[k].each_with_index do |xi, i|
                # TODO clean this up
                vc = probabilities[:conditional][k][xi] || probabilities[:conditional][k][xi.to_s]

                # unknown value if not vc
                if vc
                  denom = probabilities[:conditional][k].map { |k, v| v[c] }.sum.to_f
                  p2 = vc[c].to_f / denom

                  # TODO use proper smoothing instead
                  p2 = tiny_p if p2 == 0

                  px[i] += Math.log(p2)
                end
              end
            when "derived"
              @derived[k].each do |k2, v2|
                vc = probabilities[:conditional][k2][c]

                x.columns[k].each_with_index do |xi, i|
                  px[i] += Math.log(calculate_probability(xi == v2 ? 1 : 0, vc[:mean], vc[:stdev]))
                end
              end
            else
              vc = probabilities[:conditional][k][c]

              if vc[:stdev] != 0 && !vc[:stdev].nil?
                x.columns[k].each_with_index do |xi, i|
                  px[i] += Math.log(calculate_probability(xi, vc[:mean], vc[:stdev]))
                end
              else
                x.columns[k].each_with_index do |xi, i|
                  if xi != vc[:mean]
                    # TODO use proper smoothing instead
                    px[i] += Math.log(tiny_p)
                  end
                end
              end
            end

            probs.columns[c] = px
          end
        end

        probs
      end

      SQRT_2PI = Math.sqrt(2 * Math::PI)

      # TODO memoize for performance
      def calculate_probability(x, mean, stdev)
        exponent = Math.exp(-((x - mean)**2) / (2 * (stdev**2)))
        (1 / (SQRT_2PI * stdev)) * exponent
      end
    end
  end
end
