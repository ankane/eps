module Eps
  class NaiveBayes < BaseEstimator
    attr_reader :probabilities

    def accuracy
      Eps::Metrics.accuracy(@train_set.label, predict(@train_set))
    end

    # pmml

    def self.load_pmml(data)
      super do |data|
        # TODO more validation
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
                prob[n3.attribute("value").value][n2.attribute("value").value] = BigDecimal(n3.attribute("count").value)
              end
            else
              boom = {}
              n2.css("TargetValueCount").each do |n3|
                boom[n3.attribute("value").value] = BigDecimal(n3.attribute("count").value)
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
    end

    private

    # TODO better summary
    def _summary(extended: false)
      str = String.new("")
      probabilities[:prior].each do |k, v|
        str += "#{k}: #{v}\n"
      end
      str
    end

    def _train(smoothing: 1, **options)
      raise "Target must be strings" if @target_type != "categorical"
      check_missing_value(@train_set)
      check_missing_value(@validation_set) if @validation_set
      raise ArgumentError, "weight not supported" if @train_set.weight

      data = @train_set

      prep_text_features(data)

      # convert boolean to strings
      data.label = data.label.map(&:to_s)

      indexes = {}
      data.label.each_with_index do |yi, i|
        (indexes[yi] ||= []) << i
      end

      grouped = {}
      indexes.each do |k, v|
        grouped[k] = data[v]
      end

      prior = {}
      grouped.sort_by { |k, _| k }.each do |k, v|
        prior[k] = v.size
      end
      labels = prior.keys

      target_counts = {}
      labels.each do |k|
        target_counts[k] = 0
      end

      conditional = {}

      @features.each do |k, type|
        prob = {}

        case type
        when "text"
          raise "Text features not supported yet for naive Bayes"
        when "categorical"
          groups = Hash.new { |hash, key| hash[key] = [] }
          data.columns[k].each_with_index do |v, i|
            groups[v] << i
          end

          groups.each do |group, indexes|
            df = data[indexes]
            prob[group] = group_count(df.label, target_counts.dup)
          end

          # smooth
          if smoothing
            labels.each do |label|
              sum = prob.map { |k2, v2| v2[label] }.sum.to_f
              prob.each do |k2, v|
                v[label] = (v[label] + smoothing) * sum / (sum + (prob.size * smoothing))
              end
            end
          end
        else
          labels.each do |group|
            xs = grouped[group]

            # TODO handle this case
            next unless xs

            values = xs.columns[k]
            prob[group] = {mean: mean(values), stdev: stdev(values)}
          end
        end

        conditional[k] = prob
      end

      @probabilities = {
        prior: prior,
        conditional: conditional
      }

      Evaluators::NaiveBayes.new(probabilities: probabilities, features: @features)
    end

    def generate_pmml
      data_fields = {}
      data_fields[@target] = probabilities[:prior].keys
      probabilities[:conditional].each do |k, v|
        if @features[k] == "categorical"
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
                if @features[k] == "categorical"
                  v.sort_by { |k2, _| k2 }.each do |k2, v2|
                    xml.PairCounts(value: k2) do
                      xml.TargetValueCounts do
                        v2.sort_by { |k2, _| k2 }.each do |k3, v3|
                          xml.TargetValueCount(value: k3, count: v3)
                        end
                      end
                    end
                  end
                else
                  xml.TargetValueStats do
                    v.sort_by { |k2, _| k2 }.each do |k2, v2|
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
              probabilities[:prior].sort_by { |k, _| k }.each do |k, v|
                xml.TargetValueCount(value: k, count: v)
              end
            end
          end
        end
      end
    end

    def group_count(arr, start)
      arr.inject(start) { |h, e| h[e] += 1; h }
    end

    def mean(arr)
      arr.sum / arr.size.to_f
    end

    def stdev(arr)
      return nil if arr.size <= 1
      m = mean(arr)
      sum = arr.inject(0) { |accum, i| accum + (i - m)**2 }
      Math.sqrt(sum / (arr.length - 1).to_f)
    end
  end
end
