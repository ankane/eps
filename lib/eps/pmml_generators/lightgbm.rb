module Eps
  module PmmlGenerators
    module LightGBM
      private

      def generate_pmml
        feature_importance = @feature_importance

        data_fields = {}
        data_fields[@target] = @labels if @labels
        @features.each_with_index do |(k, type), i|
          # TODO remove zero importance features
          if type == "categorical"
            data_fields[k] = @label_encoders[k].labels.keys
          else
            data_fields[k] = nil
          end
        end

        build_pmml(data_fields) do |xml|
          function_name = @objective == "regression" ? "regression" : "classification"
          xml.MiningModel(functionName: function_name, algorithmName: "LightGBM") do
            xml.MiningSchema do
              xml.MiningField(name: @target, usageType: "target")
              @features.keys.each_with_index do |k, i|
                # next if feature_importance[i] == 0
                # TODO add importance, but need to handle text features
                xml.MiningField(name: k) #, importance: feature_importance[i].to_f, missingValueTreatment: "asIs")
              end
            end
            pmml_local_transformations(xml)

            case @objective
            when "regression"
              xml_segmentation(xml, @trees)
            when "binary"
              xml.Segmentation(multipleModelMethod: "modelChain") do
                xml.Segment(id: 1) do
                  xml.True
                  xml.MiningModel(functionName: "regression") do
                    xml.MiningSchema do
                      @features.each do |k, _|
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
                    xml_segmentation(xml, @trees)
                  end
                end
                xml.Segment(id: 2) do
                  xml.True
                  xml.RegressionModel(functionName: "classification", normalizationMethod: "none") do
                    xml.MiningSchema do
                      xml.MiningField(name: @target, usageType: "target")
                      xml.MiningField(name: "transformedLgbmValue")
                    end
                    xml.Output do
                      @labels.each do |label|
                        xml.OutputField(name: "probability(#{label})", optype: "continuous", dataType: "double", feature: "probability", value: label)
                      end
                    end
                    xml.RegressionTable(intercept: 0.0, targetCategory: @labels.last) do
                      xml.NumericPredictor(name: "transformedLgbmValue", coefficient: "1.0")
                    end
                    xml.RegressionTable(intercept: 0.0, targetCategory: @labels.first)
                  end
                end
              end
            else # multiclass
              xml.Segmentation(multipleModelMethod: "modelChain") do
                n = @trees.size / @labels.size
                @trees.each_slice(n).each_with_index do |trees, idx|
                  xml.Segment(id: idx + 1) do
                    xml.True
                    xml.MiningModel(functionName: "regression") do
                      xml.MiningSchema do
                        @features.each do |k, _|
                          xml.MiningField(name: k)
                        end
                      end
                      xml.Output do
                        xml.OutputField(name: "lgbmValue(#{@labels[idx]})", optype: "continuous", dataType: "double", feature: "predictedValue", isFinalResult: false)
                      end
                      xml_segmentation(xml, trees)
                    end
                  end
                end
                xml.Segment(id: @labels.size + 1) do
                  xml.True
                  xml.RegressionModel(functionName: "classification", normalizationMethod: "softmax") do
                    xml.MiningSchema do
                      xml.MiningField(name: @target, usageType: "target")
                      @labels.each do |label|
                        xml.MiningField(name: "lgbmValue(#{label})")
                      end
                    end
                    xml.Output do
                      @labels.each do |label|
                        xml.OutputField(name: "probability(#{label})", optype: "continuous", dataType: "double", feature: "probability", value: label)
                      end
                    end
                    @labels.each do |label|
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
    end
  end
end
