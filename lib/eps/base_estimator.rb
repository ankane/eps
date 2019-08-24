module Eps
  class BaseEstimator
    def train(data, y = nil, target: nil, split: false, validation_set: nil, verbose: nil, **options)
      data, @target = prep_data(data, y, target)
      @target_type = Utils.column_type(data.label, @target)

      # determine feature types
      @features = {}
      data.columns.each do |k, v|
        @features[k] = Utils.column_type(v, k)
      end

      # cross validation
      if split && !validation_set
        split_p = 0.7
        if split == true
          rng = Random.new(0) # seed random number generator
          train_idx, test_idx = (0...data.size).to_a.partition { rng.rand < split_p }
        else
          split_column = split.to_s
          times = data.columns.delete(split_column)
          check_missing(times, split_column)
          split_index = (times.size * split_p).round
          split_time = times.sort[split_index]
          train_idx, test_idx = (0...data.size).to_a.partition { |i| times[i] < split_time }
        end
        @train_set = data[train_idx]
        validation_set = data[test_idx]
      else
        @train_set = data
        if validation_set
          validation_set = Eps::DataFrame.new(validation_set)
          validation_set.label = validation_set.columns.delete(@target)
        end
      end

      @evaluator = _train

      # reset pmml
      @pmml = nil

      # summary
      if verbose != false
        if validation_set
          y_true = validation_set.label
          y_pred = predict(validation_set)

          case @target_type
          when "numeric"
            metric_name = "RMSE"
            v = Metrics.rmse(y_true, y_pred)
            metric_value = v.round >= 1000 ? v.round.to_s : "%.3g" % v
          else
            metric_name = "accuracy"
            metric_value = "%.1f%%" % (100 * Metrics.accuracy(y_true, y_pred)).round(1)
          end

          puts "Validation %s: %s"  % [metric_name, metric_value]
          puts
        end
        puts summary
      end

      nil
    end

    def predict(data)
      singular = data.is_a?(Hash)
      data = [data] if singular

      predictions = @evaluator.predict(Eps::DataFrame.new(data))

      singular ? predictions.first : predictions
    end

    def evaluate(data, y = nil, target: nil)
      data, target = prep_data(data, y, target || @target)
      Eps.metrics(data.label, predict(data))
    end

    def to_pmml
      (@pmml ||= generate_pmml).to_xml
    end

    def self.load_pmml(data)
      if data.is_a?(String)
        data = Nokogiri::XML(data) { |config| config.strict }
      end
      model = new
      model.instance_variable_set("@pmml", data) # cache data
      model.instance_variable_set("@evaluator", yield(data))
      model
    end

    private

    def prep_data(data, y, target)
      data = Eps::DataFrame.new(data)
      target = (target || "target").to_s
      data.label = (y || data.columns.delete(target)).to_a
      check_data(data)
      [data, target]
    end

    def check_data(data)
      raise "No data" if data.empty?
      raise "Number of samples differs from target" if data.size != data.label.size
      raise "Target missing in data" if data.label.any?(&:nil?)
    end

    # pmml

    def build_pmml(data_fields)
      Nokogiri::XML::Builder.new do |xml|
        xml.PMML(version: "4.4", xmlns: "http://www.dmg.org/PMML-4_4", "xmlns:xsi" => "http://www.w3.org/2001/XMLSchema-instance") do
          pmml_header(xml)
          pmml_data_dictionary(xml, data_fields)
          yield xml
        end
      end
    end

    def pmml_header(xml)
      xml.Header do
        xml.Application(name: "Eps", version: Eps::VERSION)
        xml.Timestamp Time.now.utc.iso8601
      end
    end

    def pmml_data_dictionary(xml, data_fields)
      xml.DataDictionary do
        @features.each do |k, type|
          if type == "categorical"
            xml.DataField(name: k, optype: "categorical", dataType: "string") do
              data_fields[k].map(&:to_s).sort.each do |v|
                xml.Value(value: v)
              end
            end
          else
            xml.DataField(name: k, optype: "continuous", dataType: "double")
          end
        end
      end
    end
  end
end
