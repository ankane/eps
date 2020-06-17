module Eps
  class BaseEstimator
    def initialize(data = nil, y = nil, **options)
      @options = options.dup
      @trained = false
      @text_encoders = {}
      # TODO better pattern - don't pass most options to train
      train(data, y, **options) if data
    end

    def predict(data)
      _predict(data, false)
    end

    def predict_probability(data)
      _predict(data, true)
    end

    def evaluate(data, y = nil, target: nil, weight: nil)
      data, target = prep_data(data, y, target || @target, weight)
      Eps.metrics(data.label, predict(data), weight: data.weight)
    end

    def to_pmml
      @pmml ||= PMML.generate(self)
    end

    def self.load_pmml(pmml)
      model = new
      model.instance_variable_set("@evaluator", PMML.load(pmml))
      model.instance_variable_set("@pmml", pmml.respond_to?(:to_xml) ? pmml.to_xml : pmml) # cache data
      model
    end

    def summary(extended: false)
      raise "Summary not available for loaded models" unless @trained

      str = String.new("")

      if @validation_set
        y_true = @validation_set.label
        y_pred = predict(@validation_set)

        case @target_type
        when "numeric"
          metric_name = "RMSE"
          v = Metrics.rmse(y_true, y_pred, weight: @validation_set.weight)
          metric_value = v.round >= 1000 ? v.round.to_s : "%.3g" % v
        else
          metric_name = "accuracy"
          metric_value = "%.1f%%" % (100 * Metrics.accuracy(y_true, y_pred, weight: @validation_set.weight)).round(1)
        end
        str << "Validation %s: %s\n\n"  % [metric_name, metric_value]
      end

      str << _summary(extended: extended)
      str
    end

    private

    def _predict(data, probabilities)
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

      predictions = @evaluator.predict(data, probabilities: probabilities)

      singular ? predictions.first : predictions
    end

    def train(data, y = nil, target: nil, weight: nil, split: nil, validation_set: nil, text_features: nil, **options)
      data, @target = prep_data(data, y, target, weight)
      @target_type = Utils.column_type(data.label, @target)

      if split.nil?
        split = data.size >= 30
      end

      # cross validation
      # TODO adjust based on weight
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
          raise "Target required for validation set" unless target
          raise "Weight required for validation set" if data.weight && !weight
          validation_set, _ = prep_data(validation_set, nil, @target, weight)
        end
      end

      raise "No data in training set" if @train_set.empty?
      raise "No data in validation set" if validation_set && validation_set.empty?

      @validation_set = validation_set
      @evaluator = _train(**options)

      # reset pmml
      @pmml = nil

      @trained = true

      nil
    end

    def prep_data(data, y, target, weight)
      data = Eps::DataFrame.new(data)

      # target
      target = (target || "target").to_s
      y ||= data.columns.delete(target)
      check_missing(y, target)
      data.label = y.to_a

      # weight
      if weight
        weight =
          if weight.respond_to?(:to_a)
            weight.to_a
          else
            data.columns.delete(weight.to_s)
          end
        check_missing(weight, "weight")
        data.weight = weight.to_a
      end

      check_data(data)
      [data, target]
    end

    def prep_text_features(train_set, fit: true)
      @text_features.each do |k, v|
        if fit
          # reset vocabulary
          v.delete(:vocabulary)

          # TODO determine max features automatically
          # start based on number of rows
          encoder = Eps::TextEncoder.new(**v)
          counts = encoder.fit(train_set.columns.delete(k))
        else
          encoder = @text_encoders[k]
          counts = encoder.transform(train_set.columns.delete(k))
        end

        encoder.vocabulary.each do |word|
          train_set.columns[[k, word]] = [0] * counts.size
        end

        counts.each_with_index do |ci, i|
          ci.each do |word, count|
            word_key = [k, word]
            train_set.columns[word_key][i] = 1 if train_set.columns.key?(word_key)
          end
        end

        if fit
          @text_encoders[k] = encoder

          # update vocabulary
          v[:vocabulary] = encoder.vocabulary
        end
      end

      raise "No features left" if train_set.columns.empty?
    end

    def check_data(data)
      raise "No data" if data.empty?
      raise "Number of data points differs from target" if data.size != data.label.size
      raise "Number of data points differs from weight" if data.weight && data.size != data.weight.size
    end

    def check_missing(c, name)
      raise ArgumentError, "Missing column: #{name}" if !c
      raise ArgumentError, "Missing values in column #{name}" if c.to_a.any?(&:nil?)
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
  end
end
