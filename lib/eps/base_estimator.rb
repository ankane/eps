module Eps
  class BaseEstimator
    def train(data, y = nil, target: nil, **options)
      @data, @target = prep_data(data, y, target)
      @target_type = Utils.column_type(@data.label, @target)

      # determine feature types
      @features = {}
      @data.columns.each do |k, v|
        @features[k] = Utils.column_type(v, k)
      end
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
