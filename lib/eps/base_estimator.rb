module Eps
  class BaseEstimator
    def initialize(data = nil, y = nil, **options)
      train(data, y, **options) if data
    end

    def predict(data)
      singular = data.is_a?(Hash)
      data = [data] if singular

      data = Eps::DataFrame.new(data)

      @evaluator.features.each do |k, type|
        values = data.columns[k]
        raise ArgumentError, "Missing column: #{k}" if !values
        column_type = Utils.column_type(values.compact, k) if values

        if !column_type.nil?
          if (type == "numeric" && column_type != "numeric") || (type != "numeric" && column_type != "categorical")
            raise ArgumentError, "Bad type for column #{k}: Expected #{type} but got #{column_type}"
          end
        end
        # TODO check for unknown values for categorical features
      end

      predictions = @evaluator.predict(data)

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

    def summary(extended: false)
      str = String.new("")

      if @validation_set
        y_true = @validation_set.label
        y_pred = predict(@validation_set)

        case @target_type
        when "numeric"
          metric_name = "RMSE"
          v = Metrics.rmse(y_true, y_pred)
          metric_value = v.round >= 1000 ? v.round.to_s : "%.3g" % v
        else
          metric_name = "accuracy"
          metric_value = "%.1f%%" % (100 * Metrics.accuracy(y_true, y_pred)).round(1)
        end
        str << "Validation %s: %s\n\n"  % [metric_name, metric_value]
      end

      str << _summary(extended: extended)
      str
    end

    # private
    def self.extract_text_features(data, features)
      # updates features object
      vocabulary = {}
      function_mapping = {}
      derived_fields = {}
      data.css("LocalTransformations DerivedField, TransformationDictionary DerivedField").each do |n|
        name = n.attribute("name")&.value
        field = n.css("FieldRef").attribute("field").value
        value = n.css("Constant").text

        field = field[10..-2] if field =~ /\Alowercase\(.+\)\z/
        next if value.empty?

        (vocabulary[field] ||= []) << value

        function_mapping[field] = n.css("Apply").attribute("function").value

        derived_fields[name] = [field, value]
      end

      functions = {}
      data.css("TransformationDictionary DefineFunction").each do |n|
        name = n.attribute("name").value
        text_index = n.css("TextIndex")
        functions[name] = {
          tokenizer: Regexp.new(text_index.attribute("wordSeparatorCharacterRE").value),
          case_sensitive: text_index.attribute("isCaseSensitive")&.value == "true"
        }
      end

      text_features = {}
      function_mapping.each do |field, function|
        text_features[field] = functions[function].merge(vocabulary: vocabulary[field])
        features[field] = "text"
      end

      [text_features, derived_fields]
    end

    private

    def train(data, y = nil, target: nil, split: nil, validation_set: nil, verbose: nil, text_features: nil, early_stopping: nil)
      data, @target = prep_data(data, y, target)
      @target_type = Utils.column_type(data.label, @target)

      if split.nil?
        split = data.size >= 30
      end

      # cross validation
      if split && !validation_set
        split = {} if split == true
        split = {column: split} unless split.is_a?(Hash)

        split_p = 1 - (split[:validation_size] || 0.25)
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
        @features[k] = Utils.column_type(v.compact, k)
      end

      # determine text features if not specified
      if text_features.nil?
        text_features = []

        @features.each do |k, type|
          next if type != "categorical"

          values = data.columns[k].compact

          next unless values.first.is_a?(String) # not boolean

          values = values.reject(&:empty?)
          count = values.count

          # check if spaces
          # two spaces is rough approximation for 3 words
          # TODO make more performant
          if values.count { |v| v.count(" ") >= 2 } > 0.5 * count
            text_features << k
          end
        end
      end

      # prep text features
      @text_features = {}
      (text_features || {}).each do |k, v|
        @features[k.to_s] = "text"

        # same output as scikit-learn CountVectorizer
        # except for max_features
        @text_features[k.to_s] = {
          tokenizer: /\W+/,
          min_length: 2,
          max_features: 100
        }.merge(v || {})
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

      @validation_set = validation_set
      @evaluator = _train(verbose: verbose, early_stopping: early_stopping)

      # reset pmml
      @pmml = nil

      nil
    end

    def prep_data(data, y, target)
      data = Eps::DataFrame.new(data)
      target = (target || "target").to_s
      y ||= data.columns.delete(target)
      check_missing(y, target)
      data.label = y.to_a
      check_data(data)
      [data, target]
    end

    def prep_text_features(train_set)
      @text_encoders = {}
      @text_features.each do |k, v|
        # reset vocabulary
        v.delete(:vocabulary)

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

        # update vocabulary
        v[:vocabulary] = encoder.vocabulary
      end

      raise "No features left" if train_set.columns.empty?
    end

    def check_data(data)
      raise "No data" if data.empty?
      raise "Number of data points differs from target" if data.size != data.label.size
    end

    def check_missing(c, name)
      raise ArgumentError, "Missing column: #{name}" if !c
      raise ArgumentError, "Missing values in column #{name}" if c.any?(&:nil?)
    end

    def check_missing_value(df)
      df.columns.each do |k, v|
        check_missing(v, k)
      end
    end

    def display_field(k)
      if k.is_a?(Array)
        if @features[k.first] == "text"
          "#{k.first}(#{k.last})"
        else
          k.join("=")
        end
      else
        k
      end
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
        # xml.Timestamp Time.now.utc.iso8601
      end
    end

    def pmml_data_dictionary(xml, data_fields)
      xml.DataDictionary do
        data_fields.each do |k, vs|
          case @features[k]
          when "categorical", nil
            xml.DataField(name: k, optype: "categorical", dataType: "string") do
              vs.map(&:to_s).sort.each do |v|
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
              xml.DerivedField(name: display_field([k, v]), optype: "continuous", dataType: "integer") do
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
