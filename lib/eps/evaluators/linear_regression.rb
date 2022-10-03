module Eps
  module Evaluators
    class LinearRegression
      attr_reader :features

      def initialize(coefficients:, features:, text_features:)
        @coefficients = coefficients.to_h { |k, v| [k.is_a?(Array) ? [k[0].to_s, k[1]] : k.to_s, v] }
        @features = features
        @text_features = text_features || {}
      end

      def predict(x, probabilities: false)
        raise "Probabilities not supported" if probabilities

        intercept = @coefficients["_intercept"] || 0.0
        scores = [intercept] * x.size

        @features.each do |k, type|
          raise "Missing data in #{k}" if !x.columns[k] || x.columns[k].any?(&:nil?)

          case type
          when "categorical"
            x.columns[k].each_with_index do |xv, i|
              # TODO clean up
              scores[i] += (@coefficients[[k, xv]] || @coefficients[[k, xv.to_s]]).to_f
            end
          when "text"
            encoder = TextEncoder.new(**@text_features[k])
            counts = encoder.transform(x.columns[k])
            coef = {}
            @coefficients.each do |k2, v|
              next unless k2.is_a?(Array) && k2.first == k
              coef[k2.last] = v
            end

            counts.each_with_index do |xc, i|
              xc.each do |word, count|
                scores[i] += coef[word] * count if coef[word]
              end
            end
          else
            coef = @coefficients[k].to_f
            x.columns[k].each_with_index do |xv, i|
              scores[i] += coef * xv
            end
          end
        end

        scores
      end

      def coefficients
        @coefficients.to_h { |k, v| [Array(k).join.to_sym, v] }
      end
    end
  end
end
