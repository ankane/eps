module Eps
  class Model
    def initialize(data = nil, y = nil, estimator: nil, **options)
      if estimator
        @estimator = estimator
      elsif data
        train(data, y, **options)
      end
    end

    # pmml

    def self.load_pmml(data)
      if data.is_a?(String)
        data = Nokogiri::XML(data) { |config| config.strict }
      end

      estimator_class =
        if data.css("Segmentation").any?
          LightGBM
        elsif data.css("RegressionModel").any?
          LinearRegression
        elsif data.css("NaiveBayesModel").any?
          NaiveBayes
        else
          raise "Unknown model"
        end

      new(estimator: estimator_class.load_pmml(data))
    end

    private

    def train(data, y = nil, target: nil, algorithm: :lightgbm, **options)
      estimator_class =
        case algorithm
        when :lightgbm
          LightGBM
        when :linear_regression
          LinearRegression
        when :naive_bayes
          NaiveBayes
        else
          raise ArgumentError, "Unknown algorithm: #{algorithm}"
        end

      @estimator = estimator_class.new(data, y, target: target, **options)
    end

    def respond_to_missing?(name, include_private = false)
      if @estimator
        @estimator.respond_to?(name, include_private)
      else
        super
      end
    end

    def method_missing(method, *args, &block)
      if @estimator && @estimator.respond_to?(method)
        @estimator.public_send(method, *args, &block)
      else
        super
      end
    end
  end
end
