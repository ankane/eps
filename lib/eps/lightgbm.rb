module Eps
  class LightGBM < BaseEstimator
    private

    def _summary(extended: false)
      str = String.new("")
      importance = @booster.feature_importance
      total = importance.sum.to_f
      if total == 0
        str << "Model needs more data for better predictions\n"
      else
        str << "Most important features\n"
        @importance_keys.zip(importance).sort_by { |k, v| [-v, display_field(k)] }.first(10).each do |k, v|
          str << "#{display_field(k)}: #{(100 * v / total).round}\n"
        end
      end
      str
    end

    def _train(verbose: nil, early_stopping: nil, learning_rate: 0.1)
      train_set = @train_set
      validation_set = @validation_set.dup
      summary_label = train_set.label

      # create check set
      evaluator_set = validation_set || train_set
      check_idx = 100.times.map { rand(evaluator_set.size) }.uniq
      evaluator_set = evaluator_set[check_idx]

      # objective
      objective =
        if @target_type == "numeric"
          "regression"
        else
          label_encoder = LabelEncoder.new
          train_set.label = label_encoder.fit_transform(train_set.label)
          validation_set.label = label_encoder.transform(validation_set.label) if validation_set
          labels = label_encoder.labels.keys

          if labels.size > 2
            "multiclass"
          else
            "binary"
          end
        end

      # label encoding
      label_encoders = {}
      @features.each do |k, type|
        if type == "categorical"
          label_encoder = LabelEncoder.new
          train_set.columns[k] = label_encoder.fit_transform(train_set.columns[k])
          validation_set.columns[k] = label_encoder.transform(validation_set.columns[k]) if validation_set
          label_encoders[k] = label_encoder
        end
      end

      # text feature encoding
      prep_text_features(train_set)
      prep_text_features(validation_set, fit: false) if validation_set

      # create params
      params = {
        objective: objective,
        learning_rate: learning_rate
      }
      params[:num_classes] = labels.size if objective == "multiclass"
      if train_set.size < 30
        params[:min_data_in_bin] = 1
        params[:min_data_in_leaf] = 1
      end

      # create datasets
      categorical_idx = train_set.columns.keys.map.with_index.select { |k, _| @features[k] == "categorical" }.map(&:last)
      train_ds = ::LightGBM::Dataset.new(train_set.map_rows(&:to_a), label: train_set.label, weight: train_set.weight, categorical_feature: categorical_idx, params: params)
      validation_ds = ::LightGBM::Dataset.new(validation_set.map_rows(&:to_a), label: validation_set.label, weight: validation_set.weight, categorical_feature: categorical_idx, params: params, reference: train_ds) if validation_set

      # train
      valid_sets = [train_ds]
      valid_sets << validation_ds if validation_ds
      valid_names = ["training"]
      valid_names << "validation" if validation_ds
      early_stopping = 50 if early_stopping.nil? && validation_ds
      early_stopping = nil if early_stopping == false
      booster = ::LightGBM.train(params, train_ds, num_boost_round: 1000, early_stopping_rounds: early_stopping, valid_sets: valid_sets, valid_names: valid_names, verbose_eval: verbose || false)

      # separate summary from verbose_eval
      puts if verbose

      @importance_keys = train_set.columns.keys

      # create evaluator
      @label_encoders = label_encoders
      booster_tree = JSON.parse(booster.to_json)
      trees = booster_tree["tree_info"].map { |s| parse_tree(s["tree_structure"]) }
      # reshape
      if objective == "multiclass"
        new_trees = []
        grouped = trees.each_slice(labels.size).to_a
        labels.size.times do |i|
          new_trees.concat grouped.map { |v| v[i] }
        end
        trees = new_trees
      end

      # for pmml
      @objective = objective
      @labels = labels
      @feature_importance = booster.feature_importance
      @trees = trees
      @booster = booster

      # reset pmml
      @pmml = nil

      evaluator = Evaluators::LightGBM.new(trees: trees, objective: objective, labels: labels, features: @features, text_features: @text_features)
      booster_set = validation_set ? validation_set[check_idx] : train_set[check_idx]
      check_evaluator(objective, labels, booster, booster_set, evaluator, evaluator_set)
      evaluator
    end

    # compare a subset of predictions to check for possible bugs in evaluator
    # NOTE LightGBM must use double data type for prediction input for these to be consistent
    def check_evaluator(objective, labels, booster, booster_set, evaluator, evaluator_set)
      expected = @booster.predict(booster_set.map_rows(&:to_a))
      if objective == "multiclass"
        actual = evaluator.predict(evaluator_set, probabilities: true)
        # just compare first for now
        expected.map! { |v| v.first }
        actual.map! { |v| v.values.first }
      elsif objective == "binary"
        actual = evaluator.predict(evaluator_set, probabilities: true).map { |v| v.values.last }
      else
        actual = evaluator.predict(evaluator_set)
      end

      regression = objective == "regression" || objective == "binary"
      bad_observations = []
      expected.zip(actual).each_with_index do |(exp, act), i|
        success = (act - exp).abs < 0.001
        unless success
          bad_observations << {expected: exp, actual: act, data_point: evaluator_set[i].map(&:itself).first}
        end
      end

      if bad_observations.any?
        bad_observations.each do |obs|
          p obs
        end
        raise "Bug detected in evaluator. Please report an issue."
      end
    end

    # for evaluator

    def parse_tree(node)
      if node["leaf_value"]
        score = node["leaf_value"]
        Evaluators::Node.new(score: score, leaf_index: node["leaf_index"])
      else
        field = @importance_keys[node["split_feature"]]
        operator =
          case node["decision_type"]
          when "=="
            "equal"
          when "<="
            node["default_left"] ? "greaterThan" : "lessOrEqual"
          else
            raise "Unknown decision type: #{node["decision_type"]}"
          end

        value =
          if operator == "equal"
            if node["threshold"].include?("||")
              operator = "in"
              @label_encoders[field].inverse_transform(node["threshold"].split("||"))
            else
              @label_encoders[field].inverse_transform([node["threshold"]])[0]
            end
          else
            node["threshold"]
          end

        predicate = {
          field: field,
          value: value,
          operator: operator
        }

        left = parse_tree(node["left_child"])
        right = parse_tree(node["right_child"])

        if node["default_left"]
          right.predicate = predicate
          left.children.unshift right
          left
        else
          left.predicate = predicate
          right.children.unshift left
          right
        end
      end
    end
  end
end
