module Eps
  class Model
    def initialize(data = nil, y = nil, target: nil, estimator: nil, **options)
      @options = options

      if estimator
        @estimator = estimator
      elsif data # legacy
        train(data, y, target: target)
      end
    end

    # pmml

    def self.load_pmml(data)
      if data.is_a?(String)
        require "nokogiri"
        data = Nokogiri::XML(data) { |config| config.strict }
      end

      estimator_class =
        if data.css("RegressionModel").any?
          Eps::LinearRegression
        elsif data.css("NaiveBayesModel").any?
          Eps::NaiveBayes
        else
          raise "Unknown model"
        end

      new(estimator: estimator_class.load_pmml(data))
    end

    def to_pmml
      if @estimator
        require "nokogiri"
        @estimator.to_pmml
      else
        super
      end
    end

    # ruby - legacy

    def self.load(data)
      new(estimator: Eps::LinearRegression.load(data))
    end

    # json - legacy

    def self.load_json(data)
      new(estimator: Eps::LinearRegression.load_json(data))
    end

    def to_json(opts = {})
      @estimator ? @estimator.to_json(opts) : super
    end

    # pfa - legacy

    def self.load_pfa(data)
      new(estimator: Eps::LinearRegression.load_pfa(data))
    end

    # metrics

    def self.metrics(actual, estimated)
      estimator_class =
        if numeric?(actual)
          Eps::LinearRegression
        else
          Eps::NaiveBayes
        end

      estimator_class.metrics(actual, estimated)
    end

    private

    def train(data, y = nil, target: nil)
      data = Eps::DataFrame.new(data) unless data.is_a?(Eps::DataFrame)
      target = target.to_s

      y ||= data.columns[target]

      raise "Target missing in data" if !y

      estimator_class =
        if self.class.numeric?(y)
          Eps::LinearRegression
        else
          Eps::NaiveBayes
        end

      @estimator = estimator_class.new(**@options)
      @estimator.train(data, y, target: target)
    end

    def respond_to_missing?(name, include_private = false)
      if @estimator
        @estimator.respond_to?(name, include_private)
      else
        super
      end
    end

    def method_missing(method, *args, &block)
      if @estimator
        @estimator.public_send(method, *args, &block)
      else
        super
      end
    end

    def self.numeric?(y)
      y.first.is_a?(Numeric)
    end
  end
end
