module Eps
  class BaseEstimator
    def train(data, y, target: nil, **options)
      @x = normalize_x(data)
      @y = y.to_a
      @target = target || "target"

      check_data(@x, @y)

      # determine feature types
      @features = {}
      @x.columns.each do |k, v|
        @features[k] = Utils.column_type(v, k)
      end
    end

    def predict(x)
      singular = x.is_a?(Hash)
      x = [x] if singular

      x = normalize_x(x)
      pred = @evaluator.predict(x)

      singular ? pred[0] : pred
    end

    def evaluate(data, y = nil, target: nil)
      target ||= @target
      raise ArgumentError, "missing target" if !target && !y

      data = normalize_x(data)
      actual = y || data.columns[target.to_s]
      check_data(data, actual)

      estimated = predict(data)

      Eps.metrics(actual, estimated)
    end

    private

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

    def check_data(x, y)
      raise "No data" if x.empty?
      raise "Number of samples differs from target" if x.size != y.size
      raise "Target missing in data" if y.any?(&:nil?)
    end

    def normalize_x(x)
      Eps::DataFrame.new(x)
    end
  end
end
