module Eps
  module Evaluators
    class LinearRegression
      def initialize(coefficients:, features: nil, text_features: nil)
        @coefficients = Hash[coefficients.map { |k, v| [k.is_a?(Array) ? [k[0].to_s, k[1]] : k.to_s, v] }]
        @features = features
        @text_features = text_features || {}

        # legacy
        unless @features
          @features = Hash[@coefficients.keys.map { |k| [k.is_a?(Array) ? k.first : k, k.is_a?(Array) ? "categorical" : "numeric"] }]
          @features.delete("_intercept")
        end
      end

      def predict(x)
        intercept = @coefficients["_intercept"]
        scores = [intercept] * x.size

        legacy_format = false

        @features.each do |k, type|
          if !x.columns[k] && type == "numeric" && !@features.any? { |k, v| v == "categorical" }
            legacy_format = true
            expand_legacy_format(x)
          end

          raise "Missing data in #{k}" if !x.columns[k] || x.columns[k].any?(&:nil?)

          case type
          when "categorical"
            x.columns[k].each_with_index do |xv, i|
              scores[i] += @coefficients[[k, xv]].to_f
            end
          when "text"
            encoder = TextEncoder.new(@text_features[k])
            counts = encoder.fit(x.columns[k])
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

        if legacy_format
          # only warn when method completes successfully
          warn "[eps] DEPRECATION WARNING: Thanks for being an early adopter!\nUnfortunately, this model is stored in a legacy format.\nIt will stop working with Eps 0.3.0.\nPlease retrain the model and store as PMML."
        end

        scores
      end

      def coefficients
        Hash[@coefficients.map { |k, v| [Array(k).join.to_sym, v] }]
      end

      private

      # expand categorical features
      def expand_legacy_format(x)
        x.columns.keys.each do |k2|
          v2 = x.columns[k2]
          if v2[0].is_a?(String)
            v2.uniq.each do |v3|
              x.columns["#{k2}#{v3}"] = [0] * v2.size
            end
            v2.each_with_index do |v3, i|
              x.columns["#{k2}#{v3}"][i] = 1
            end
            x.columns.delete(k2)
          end
        end
      end
    end
  end
end
