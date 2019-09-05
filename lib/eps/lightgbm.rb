require "eps/pmml_generators/lightgbm"

module Eps
  class LightGBM < BaseEstimator
    include PmmlGenerators::LightGBM

    def self.load_pmml(data)
      super do |data|
        objective = data.css("MiningModel").first.attribute("functionName").value
        if objective == "classification"
          labels = data.css("RegressionModel OutputField").map { |n| n.attribute("value").value }
          objective = labels.size > 2 ? "multiclass" : "binary"
        end

        features = {}
        text_features, derived_fields = extract_text_features(data, features)
        node = data.css("DataDictionary").first
        node.css("DataField")[1..-1].to_a.each do |node|
          features[node.attribute("name").value] =
            if node.attribute("optype").value == "categorical"
              "categorical"
            else
              "numeric"
            end
        end

        trees = []
        data.css("Segmentation TreeModel").each do |tree|
          node = find_nodes(tree.css("Node").first, derived_fields)
          trees << node
        end

        Evaluators::LightGBM.new(trees: trees, objective: objective, labels: labels, features: features, text_features: text_features)
      end
    end

    private

    def _summary(extended: false)
      str = String.new("")
      importance = @booster.feature_importance
      total = importance.sum.to_f
      if total == 0
        str << "Model needs more data for better predictions\n"
      else
        str << "Most important features\n"
        @importance_keys.zip(importance).sort_by { |k, v| [-v, k] }.first(10).each do |k, v|
          str << "#{display_field(k)}: #{(100 * v / total).round}\n"
        end
      end
      str
    end

    def self.find_nodes(xml, derived_fields)
      score = BigDecimal(xml.attribute("score").value).to_f

      elements = xml.elements
      xml_predicate = elements.first

      predicate =
        if xml_predicate.name == "True"
          nil
        elsif xml_predicate.name == "SimpleSetPredicate"
          operator = "in"
          value = xml_predicate.css("Array").text.scan(/"(.+?)(?<!\\)"|(\S+)/).flatten.compact.map { |v| v.gsub('\"', '"') }
          field = xml_predicate.attribute("field").value
          field = derived_fields[field] if derived_fields[field]
          {
            field: field,
            operator: operator,
            value: value
          }
        else
          operator = xml_predicate.attribute("operator").value
          value = xml_predicate.attribute("value").value
          value = BigDecimal(value).to_f if operator == "greaterThan"
          field = xml_predicate.attribute("field").value
          field = derived_fields[field] if derived_fields[field]
          {
            field: field,
            operator: operator,
            value: value
          }
        end

      children = elements[1..-1].map { |n| find_nodes(n, derived_fields) }

      Evaluators::Node.new(score: score, predicate: predicate, children: children)
    end

    def _train(verbose: nil, early_stopping: nil)
      train_set = @train_set
      validation_set = @validation_set.dup
      summary_label = train_set.label

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
      prep_text_features(validation_set) if validation_set

      # create params
      params = {objective: objective}
      params[:num_classes] = labels.size if objective == "multiclass"
      if train_set.size < 30
        params[:min_data_in_bin] = 1
        params[:min_data_in_leaf] = 1
      end

      # create datasets
      categorical_idx = @features.values.map.with_index.select { |type, _| type == "categorical" }.map(&:last)
      train_ds = ::LightGBM::Dataset.new(train_set.map_rows(&:to_a), label: train_set.label, categorical_feature: categorical_idx, params: params)
      validation_ds = ::LightGBM::Dataset.new(validation_set.map_rows(&:to_a), label: validation_set.label, categorical_feature: categorical_idx, params: params, reference: train_ds) if validation_set

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

      Evaluators::LightGBM.new(trees: trees, objective: objective, labels: labels, features: @features, text_features: @text_features)
    end

    def evaluator_class
      PmmlLoaders::LightGBM
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
