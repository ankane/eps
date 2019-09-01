module Eps
  module Evaluators
    class NaiveBayes
      attr_reader :probabilities

      def initialize(probabilities:, features:)
        @probabilities = probabilities
        @features = features
      end

      def predict(x)
        probs = calculate_class_probabilities(x)
        probs.map do |xp|
          xp.sort_by { |k, v| [-v, k] }[0][0]
        end
      end

      # use log to prevent underflow
      # https://www.antoniomallia.it/lets-implement-a-gaussian-naive-bayes-classifier-in-python.html
      def calculate_class_probabilities(x)
        probs = Eps::DataFrame.new

        # assign very small probability if probability is 0
        tiny_p = 0.0001

        total = probabilities[:prior].values.sum.to_f
        probabilities[:prior].each do |c, cv|
          prior = Math.log(cv / total)
          px = [prior] * x.size

          @features.each do |k, type|
            vc = probabilities[:conditional][k][c]

            if type == "categorical"
              vc_sum = vc.values.sum

              x.columns[k].each_with_index do |xi, i|
                p2 = vc[xi].to_f / vc_sum

                # TODO use proper smoothing instead
                p2 = tiny_p if p2 == 0

                px[i] += Math.log(p2)
              end
            else
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
