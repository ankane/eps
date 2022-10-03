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
          # use an additional label for unseen values
          # this is only used during training for the LightGBM eval_set
          # LightGBM ignores them (only uses seen categories for predictions)
          # https://github.com/microsoft/LightGBM/issues/1936
          # the evaluator also ignores them (to be consistent with LightGBM)
          # but doesn't use this code
          @labels[yi.to_s] || @labels.size
        end
      end
    end

    def inverse_transform(y)
      inverse = @labels.map(&:reverse).to_h
      y.map do |yi|
        inverse[yi.to_i]
      end
    end
  end
end
