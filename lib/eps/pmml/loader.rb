module Eps
  module PMML
    class Loader
      attr_reader :data

      def initialize(pmml)
        if pmml.is_a?(String)
          pmml = Nokogiri::XML(pmml) { |config| config.strict }
        end
        @data = pmml
      end

      def load
        if data.css("Segmentation").any?
          lightgbm
        elsif data.css("RegressionModel").any?
          linear_regression
        elsif data.css("NaiveBayesModel").any?
          naive_bayes
        else
          raise "Unknown model"
        end
      end

      private

      def lightgbm
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

      def linear_regression
        node = data.css("RegressionTable")

        coefficients = {
          "_intercept" => node.attribute("intercept").value.to_f
        }

        features = {}

        text_features, derived_fields = extract_text_features(data, features)

        node.css("NumericPredictor").each do |n|
          name = n.attribute("name").value
          if derived_fields[name]
            name = derived_fields[name]
          else
            features[name] = "numeric"
          end
          coefficients[name] = n.attribute("coefficient").value.to_f
        end

        node.css("CategoricalPredictor").each do |n|
          name = n.attribute("name").value
          coefficients[[name, n.attribute("value").value]] = n.attribute("coefficient").value.to_f
          features[name] = "categorical"
        end

        Evaluators::LinearRegression.new(coefficients: coefficients, features: features, text_features: text_features)
      end

      def naive_bayes
        node = data.css("NaiveBayesModel")

        prior = {}
        node.css("BayesOutput TargetValueCount").each do |n|
          prior[n.attribute("value").value] = n.attribute("count").value.to_f
        end

        legacy = false

        conditional = {}
        features = {}
        node.css("BayesInput").each do |n|
          prob = {}

          # numeric
          n.css("TargetValueStat").each do |n2|
            n3 = n2.css("GaussianDistribution")
            prob[n2.attribute("value").value] = {
              mean: n3.attribute("mean").value.to_f,
              stdev: Math.sqrt(n3.attribute("variance").value.to_f)
            }
          end

          # detect bad form in Eps < 0.3
          bad_format = n.css("PairCounts").map { |n2| n2.attribute("value").value } == prior.keys

          # categorical
          n.css("PairCounts").each do |n2|
            if bad_format
              n2.css("TargetValueCount").each do |n3|
                prob[n3.attribute("value").value] ||= {}
                prob[n3.attribute("value").value][n2.attribute("value").value] = n3.attribute("count").value.to_f
              end
            else
              boom = {}
              n2.css("TargetValueCount").each do |n3|
                boom[n3.attribute("value").value] = n3.attribute("count").value.to_f
              end
              prob[n2.attribute("value").value] = boom
            end
          end

          if bad_format
            legacy = true
            prob.each do |k, v|
              prior.keys.each do |k|
                v[k] ||= 0.0
              end
            end
          end

          name = n.attribute("fieldName").value
          conditional[name] = prob
          features[name] = n.css("TargetValueStat").any? ? "numeric" : "categorical"
        end

        target = node.css("BayesOutput").attribute("fieldName").value

        probabilities = {
          prior: prior,
          conditional: conditional
        }

        # get derived fields
        derived = {}
        data.css("DerivedField").each do |n|
          name = n.attribute("name").value
          field = n.css("NormDiscrete").attribute("field").value
          value = n.css("NormDiscrete").attribute("value").value
          features.delete(name)
          features[field] = "derived"
          derived[field] ||= {}
          derived[field][name] = value
        end

        Evaluators::NaiveBayes.new(probabilities: probabilities, features: features, derived: derived, legacy: legacy)
      end

      def extract_text_features(data, features)
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

      def find_nodes(xml, derived_fields)
        score = xml.attribute("score").value.to_f

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
            value = value.to_f if operator == "greaterThan" || operator == "lessOrEqual"
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
    end
  end
end
