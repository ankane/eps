module Eps
  class Model
    def initialize(data = nil, y = nil, target: nil, estimator: nil, **options)
      @options = options

      if estimator
        @estimator = estimator
      elsif data # legacy
        train(data, y, target: target, verbose: false)
      end
    end

    def train(data, y = nil, target: nil, **options)
      data = Eps::DataFrame.new(data)
      target = (target || "target").to_s
      y ||= data.columns.delete(target)

      estimator_class =
        if Utils.column_type(y, target) == "numeric"
          Eps::LinearRegression
        else
          Eps::NaiveBayes
        end

      @estimator = estimator_class.new(**@options)
      @estimator.train(data, y, target: target, **options)
    end

    # pmml

    def self.load_pmml(data)
      if data.is_a?(String)
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
      Eps.metrics(actual, estimated)
    end

    private

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
  end
end
