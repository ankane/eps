module Eps
  class BaseEstimator
    def train(data, y = nil, target: nil, split: false, validation_set: nil, verbose: nil, text_features: nil, **options)
      data, @target = prep_data(data, y, target)
      @target_type = Utils.column_type(data.label, @target)

      # prep text features
      @text_features = {}
      (text_features || {}).each do |k, v|
        # same output as scikit-learn CountVectorizer
        # except for max_features
        @text_features[k.to_s] = {
          tokenizer: /\W+/,
          min_length: 2,
          max_features: 100
        }.merge(v || {})
      end

      # cross validation
      if split && !validation_set
        split = {} if split == true
        split = {column: split} unless split.is_a?(Hash)

        # TODO rename this option
        split_p = 1 - (split[:validation_ratio] || 0.3)
        if split[:column]
          split_column = split[:column].to_s
          times = data.columns.delete(split_column)
          check_missing(times, split_column)
          split_index = (times.size * split_p).round
          split_time = split[:value] || times.sort[split_index]
          train_idx, validation_idx = (0...data.size).to_a.partition { |i| times[i] < split_time }
        else
          if split[:shuffle] != false
            rng = Random.new(0) # seed random number generator
            train_idx, validation_idx = (0...data.size).to_a.partition { rng.rand < split_p }
          else
            split_index = (data.size * split_p).round
            train_idx, validation_idx = (0...data.size).to_a.partition { |i| i < split_index }
          end
        end
      end

      # determine feature types
      @features = {}
      data.columns.each do |k, v|
        @features[k] = @text_features.key?(k) ? "text" : Utils.column_type(v, k)
      end

      if split && !validation_set
        @train_set = data[train_idx]
        validation_set = data[validation_idx]
      else
        @train_set = data.dup
        if validation_set
          validation_set = Eps::DataFrame.new(validation_set)
          validation_set.label = validation_set.columns.delete(@target)
        end
      end

      raise "No data in training set" if @train_set.empty?
      raise "No data in validation set" if validation_set && validation_set.empty?

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

      p @evaluator
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

    def prep_text_features(train_set)
      @text_encoders = {}
      @text_features.each do |k, v|
        # TODO determine max features automatically
        # start based on number of rows
        encoder = Eps::TextEncoder.new(v)
        counts = encoder.fit(train_set.columns.delete(k))
        encoder.vocabulary.each do |word|
          train_set.columns[[k, word]] = [0] * counts.size
        end
        counts.each_with_index do |ci, i|
          ci.each do |word, count|
            word_key = [k, word]
            train_set.columns[word_key][i] = 1 if train_set.columns.key?(word_key)
          end
        end
        @text_encoders[k] = encoder
      end
    end

    def check_data(data)
      raise "No data" if data.empty?
      raise "Number of samples differs from target" if data.size != data.label.size
      raise "Target missing in data" if data.label.any?(&:nil?)
    end

    def check_missing(c, name)
      raise ArgumentError, "Missing column: #{name}" if !c
      raise ArgumentError, "Missing values in column #{name}" if c.any?(&:nil?)
    end

    # pmml

    def build_pmml(data_fields)
      Nokogiri::XML::Builder.new do |xml|
        xml.PMML(version: "4.4", xmlns: "http://www.dmg.org/PMML-4_4", "xmlns:xsi" => "http://www.w3.org/2001/XMLSchema-instance") do
          pmml_header(xml)
          pmml_data_dictionary(xml, data_fields)
          pmml_transformation_dictionary(xml)
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
          case type
          when "categorical"
            xml.DataField(name: k, optype: "categorical", dataType: "string") do
              data_fields[k].map(&:to_s).sort.each do |v|
                xml.Value(value: v)
              end
            end
          when "text"
            xml.DataField(name: k, optype: "categorical", dataType: "string")
          else
            xml.DataField(name: k, optype: "continuous", dataType: "double")
          end
        end
      end
    end

    def pmml_transformation_dictionary(xml)
      if @text_features.any?
        xml.TransformationDictionary do
          @text_features.each do |k, text_options|
            xml.DefineFunction(name: "#{k}Transform", optype: "continuous") do
              xml.ParameterField(name: "text")
              xml.ParameterField(name: "term")
              xml.TextIndex(textField: "text", localTermWeights: "termFrequency", wordSeparatorCharacterRE: text_options[:tokenizer].source, isCaseSensitive: !!text_options[:case_sensitive]) do
                xml.FieldRef(field: "term")
              end
            end
          end
        end
      end
    end

    def pmml_local_transformations(xml)
      if @text_features.any?
        xml.LocalTransformations do
          @text_features.each do |k, _|
            @text_encoders[k].vocabulary.each do |v|
              xml.DerivedField(name: "#{k}@#{v}", optype: "continuous", dataType: "integer") do
                xml.Apply(function: "#{k}Transform") do
                  xml.FieldRef(field: k)
                  xml.Constant v
                end
              end
            end
          end
        end
      end
    end
  end
end
