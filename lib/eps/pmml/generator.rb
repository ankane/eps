module Eps
  module PMML
    class Generator
      attr_reader :model

      def initialize(model)
        @model = model
      end

      def generate
        case @model
        when LightGBM
          lightgbm
        when LinearRegression
          linear_regression
        when NaiveBayes
          naive_bayes
        else
          raise "Unknown model"
        end
      end

      private

      def lightgbm
        data_fields = {}
        data_fields[target] = labels if labels
        features.each_with_index do |(k, type), i|
          # TODO remove zero importance features
          if type == "categorical"
            data_fields[k] = label_encoders[k].labels.keys
          else
            data_fields[k] = nil
          end
        end

        build_pmml(data_fields) do |xml|
          function_name = objective == "regression" ? "regression" : "classification"
          xml.MiningModel(functionName: function_name, algorithmName: "LightGBM") do
            xml.MiningSchema do
              xml.MiningField(name: target, usageType: "target")
              features.keys.each_with_index do |k, i|
                # next if feature_importance[i] == 0
                # TODO add importance, but need to handle text features
                xml.MiningField(name: k) #, importance: feature_importance[i].to_f, missingValueTreatment: "asIs")
              end
            end
            pmml_local_transformations(xml)

            case objective
            when "regression"
              xml_segmentation(xml, trees)
            when "binary"
              xml.Segmentation(multipleModelMethod: "modelChain") do
                xml.Segment(id: 1) do
                  xml.True
                  xml.MiningModel(functionName: "regression") do
                    xml.MiningSchema do
                      features.each do |k, _|
                        xml.MiningField(name: k)
                      end
                    end
                    xml.Output do
                      xml.OutputField(name: "lgbmValue", optype: "continuous", dataType: "double", feature: "predictedValue", isFinalResult: false) do
                        xml.Apply(function: "/") do
                          xml.Constant(dataType: "double") do
                            1.0
                          end
                          xml.Apply(function: "+") do
                            xml.Constant(dataType: "double") do
                              1.0
                            end
                            xml.Apply(function: "exp") do
                              xml.Apply(function: "*") do
                                xml.Constant(dataType: "double") do
                                  -1.0
                                end
                                xml.FieldRef(field: "lgbmValue")
                              end
                            end
                          end
                        end
                      end
                    end
                    xml_segmentation(xml, trees)
                  end
                end
                xml.Segment(id: 2) do
                  xml.True
                  xml.RegressionModel(functionName: "classification", normalizationMethod: "none") do
                    xml.MiningSchema do
                      xml.MiningField(name: target, usageType: "target")
                      xml.MiningField(name: "transformedLgbmValue")
                    end
                    xml.Output do
                      labels.each do |label|
                        xml.OutputField(name: "probability(#{label})", optype: "continuous", dataType: "double", feature: "probability", value: label)
                      end
                    end
                    xml.RegressionTable(intercept: 0.0, targetCategory: labels.last) do
                      xml.NumericPredictor(name: "transformedLgbmValue", coefficient: "1.0")
                    end
                    xml.RegressionTable(intercept: 0.0, targetCategory: labels.first)
                  end
                end
              end
            else # multiclass
              xml.Segmentation(multipleModelMethod: "modelChain") do
                n = trees.size / labels.size
                trees.each_slice(n).each_with_index do |trees, idx|
                  xml.Segment(id: idx + 1) do
                    xml.True
                    xml.MiningModel(functionName: "regression") do
                      xml.MiningSchema do
                        features.each do |k, _|
                          xml.MiningField(name: k)
                        end
                      end
                      xml.Output do
                        xml.OutputField(name: "lgbmValue(#{labels[idx]})", optype: "continuous", dataType: "double", feature: "predictedValue", isFinalResult: false)
                      end
                      xml_segmentation(xml, trees)
                    end
                  end
                end
                xml.Segment(id: labels.size + 1) do
                  xml.True
                  xml.RegressionModel(functionName: "classification", normalizationMethod: "softmax") do
                    xml.MiningSchema do
                      xml.MiningField(name: target, usageType: "target")
                      labels.each do |label|
                        xml.MiningField(name: "lgbmValue(#{label})")
                      end
                    end
                    xml.Output do
                      labels.each do |label|
                        xml.OutputField(name: "probability(#{label})", optype: "continuous", dataType: "double", feature: "probability", value: label)
                      end
                    end
                    labels.each do |label|
                      xml.RegressionTable(intercept: 0.0, targetCategory: label) do
                        xml.NumericPredictor(name: "lgbmValue(#{label})", coefficient: "1.0")
                      end
                    end
                  end
                end
              end
            end
          end
        end
      end

      def linear_regression
        predictors = model.instance_variable_get("@coefficients").dup
        intercept = predictors.delete("_intercept") || 0.0

        data_fields = {}
        features.each do |k, type|
          if type == "categorical"
            data_fields[k] = predictors.keys.select { |k, v| k.is_a?(Array) && k.first == k }.map(&:last)
          else
            data_fields[k] = nil
          end
        end

        build_pmml(data_fields) do |xml|
          xml.RegressionModel(functionName: "regression") do
            xml.MiningSchema do
              features.each do |k, _|
                xml.MiningField(name: k)
              end
            end
            pmml_local_transformations(xml)
            xml.RegressionTable(intercept: intercept) do
              predictors.each do |k, v|
                if k.is_a?(Array)
                  if features[k.first] == "text"
                    xml.NumericPredictor(name: display_field(k), coefficient: v)
                  else
                    xml.CategoricalPredictor(name: k[0], value: k[1], coefficient: v)
                  end
                else
                  xml.NumericPredictor(name: k, coefficient: v)
                end
              end
            end
          end
        end
      end

      def naive_bayes
        data_fields = {}
        data_fields[target] = probabilities[:prior].keys
        probabilities[:conditional].each do |k, v|
          if features[k] == "categorical"
            data_fields[k] = v.keys
          else
            data_fields[k] = nil
          end
        end

        build_pmml(data_fields) do |xml|
          xml.NaiveBayesModel(functionName: "classification", threshold: 0.001) do
            xml.MiningSchema do
              data_fields.each do |k, _|
                xml.MiningField(name: k)
              end
            end
            xml.BayesInputs do
              probabilities[:conditional].each do |k, v|
                xml.BayesInput(fieldName: k) do
                  if features[k] == "categorical"
                    v.sort_by { |k2, _| k2.to_s }.each do |k2, v2|
                      xml.PairCounts(value: k2) do
                        xml.TargetValueCounts do
                          v2.sort_by { |k2, _| k2.to_s }.each do |k3, v3|
                            xml.TargetValueCount(value: k3, count: v3)
                          end
                        end
                      end
                    end
                  else
                    xml.TargetValueStats do
                      v.sort_by { |k2, _| k2.to_s }.each do |k2, v2|
                        xml.TargetValueStat(value: k2) do
                          xml.GaussianDistribution(mean: v2[:mean], variance: v2[:stdev]**2)
                        end
                      end
                    end
                  end
                end
              end
            end
            xml.BayesOutput(fieldName: "target") do
              xml.TargetValueCounts do
                probabilities[:prior].sort_by { |k, _| k.to_s }.each do |k, v|
                  xml.TargetValueCount(value: k, count: v)
                end
              end
            end
          end
        end
      end

      def display_field(k)
        if k.is_a?(Array)
          if features[k.first] == "text"
            "#{k.first}(#{k.last})"
          else
            k.join("=")
          end
        else
          k
        end
      end

      def xml_segmentation(xml, trees)
        xml.Segmentation(multipleModelMethod: "sum") do
          trees.each_with_index do |node, i|
            xml.Segment(id: i + 1) do
              xml.True
              xml.TreeModel(functionName: "regression", missingValueStrategy: "none", noTrueChildStrategy: "returnLastPrediction", splitCharacteristic: "multiSplit") do
                xml.MiningSchema do
                  node_fields(node).uniq.each do |k|
                    xml.MiningField(name: display_field(k))
                  end
                end
                node_pmml(node, xml)
              end
            end
          end
        end
      end

      def node_fields(node)
        fields = []
        fields << node.field if node.predicate
        node.children.each do |n|
          fields.concat(node_fields(n))
        end
        fields
      end

      def node_pmml(node, xml)
        xml.Node(score: node.score) do
          if node.predicate.nil?
            xml.True
          elsif node.operator == "in"
            xml.SimpleSetPredicate(field: display_field(node.field), booleanOperator: "isIn") do
              xml.Array(type: "string") do
                xml.text node.value.map { |v| escape_element(v) }.join(" ")
              end
            end
          else
            xml.SimplePredicate(field: display_field(node.field), operator: node.operator, value: node.value)
          end
          node.children.each do |n|
            node_pmml(n, xml)
          end
        end
      end

      def escape_element(v)
        "\"#{v.gsub("\"", "\\\"")}\""
      end

      def build_pmml(data_fields)
        Nokogiri::XML::Builder.new do |xml|
          xml.PMML(version: "4.4", xmlns: "http://www.dmg.org/PMML-4_4", "xmlns:xsi" => "http://www.w3.org/2001/XMLSchema-instance") do
            pmml_header(xml)
            pmml_data_dictionary(xml, data_fields)
            pmml_transformation_dictionary(xml)
            yield xml
          end
        end.to_xml
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
            case features[k]
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
        if text_features.any?
          xml.TransformationDictionary do
            text_features.each do |k, text_options|
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
        if text_features.any?
          xml.LocalTransformations do
            text_features.each do |k, _|
              text_encoders[k].vocabulary.each do |v|
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

      # TODO create instance methods on model for all of these features

      def features
        model.instance_variable_get("@features")
      end

      def text_features
        model.instance_variable_get("@text_features")
      end

      def text_encoders
        model.instance_variable_get("@text_encoders")
      end

      def feature_importance
        model.instance_variable_get("@feature_importance")
      end

      def labels
        model.instance_variable_get("@labels")
      end

      def trees
        model.instance_variable_get("@trees")
      end

      def target
        model.instance_variable_get("@target")
      end

      def label_encoders
        model.instance_variable_get("@label_encoders")
      end

      def objective
        model.instance_variable_get("@objective")
      end

      def probabilities
        model.instance_variable_get("@probabilities")
      end

      # end TODO
    end
  end
end
