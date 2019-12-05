module Eps
  class LabelEncoder
    attr_reader :labels

    def initialize
      @labels = {}
    end

    def fit(y)
      labels = {}
      y.compact.map(&:to_s).uniq.sort.each_with_index do |label, i|
        labels[label] = i
      end
      @labels = labels
    end

    def fit_transform(y)
      fit(y)
      transform(y)
    end

    def transform(y)
      y.map do |yi|
        if yi.nil?
          nil
        else
          v = @labels[yi.to_s]
          raise "Previously unseen label: #{yi}" unless v
          v
        end
      end
    end

    def inverse_transform(y)
      inverse = Hash[@labels.map(&:reverse)]
      y.map do |yi|
        inverse[yi.to_i]
      end
    end
  end
end
