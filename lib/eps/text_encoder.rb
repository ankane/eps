module Eps
  class TextEncoder
    attr_reader :options, :vocabulary

    def initialize(**options)
      @options = options
      @vocabulary = options[:vocabulary] || []
    end

    def fit(arr)
      counts, fit = count_and_fit(arr)

      min_length = options[:min_length]
      if min_length
        counts.select! { |k, _| k.length >= min_length }
      end

      min_occurrences = options[:min_occurrences]
      if min_occurrences
        counts.select! { |_, v| v >= min_occurrences }
      end

      max_occurrences = options[:max_occurrences]
      if max_occurrences
        counts.reject! { |_, v| v > max_occurrences }
      end

      max_features = options[:max_features]
      if max_features
        counts = counts.sort_by { |_, v| -v }[0...max_features].to_h
      end

      @vocabulary = counts.keys

      fit
    end

    def transform(arr)
      counts, fit = count_and_fit(arr)
      fit
    end

    private

    def count_and_fit(arr)
      tokenizer = options[:tokenizer]
      stop_words = Array(options[:stop_words])

      fit =
        arr.map do |xi|
          # tokenize
          tokens = xi.to_s
          tokens = tokens.downcase unless options[:case_sensitive]
          tokens = tokens.split(tokenizer)

          # remove stop words
          tokens -= stop_words

          # count
          xc = Hash.new(0)
          tokens.each do |token|
            xc[token] += 1
          end
          xc
        end

      counts = Hash.new(0)

      fit.each do |xc|
        xc.each do |k2, v2|
          counts[k2] += v2
        end
      end

      # remove empty strings
      counts.delete("")

      [counts, fit]
    end
  end
end
