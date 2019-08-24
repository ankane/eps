module Eps
  class NaiveBayes < BaseEstimator
    attr_reader :probabilities

    def initialize(evaluator: nil)
      @evaluator = evaluator
    end

    def train(*args)
      super

      @y = @y.map { |yi| yi.to_s }
      x = @x

      x.columns[@target] = @y

      indexes = {}
      @y.each_with_index do |yi, i|
        (indexes[yi] ||= []) << i
      end

      grouped = {}
      indexes.each do |k, v|
        grouped[k] = x[v]
      end

      prior = {}
      grouped.each do |k, v|
        prior[k] = v.size
      end

      conditional = {}
      @features.each do |k, type|
        prob = {}

        prior.keys.each do |group|
          xs = grouped[group]

          # TODO handle this case
          next unless xs

          values = xs.columns[k]

          prob[group] =
            if type == "categorical"
              # TODO apply smoothing
              # apply smoothing only to
              # 1. categorical features
              # 2. conditional probabilities
              # TODO more efficient count
              group_count(values)
            else
              {mean: mean(values), stdev: stdev(values)}
            end
        end

        conditional[k] = prob
      end

      @probabilities = {
        prior: prior,
        conditional: conditional
      }

      @evaluator = Evaluators::NaiveBayes.new(probabilities: probabilities, features: @features)
    end

    # TODO better summary
    def summary(extended: false)
      str = String.new("")
      probabilities[:prior].each do |k, v|
        str += "#{k}: #{v}\n"
      end
      str += "\n"
      str += "accuracy: %d%%\n" % [(100 * accuracy).round]
      str
    end

    def accuracy
      Eps::Metrics.accuracy(predict(@x), @y)
    end

    # pmml

    def self.load_pmml(data)
      # TODO more validation
      node = data.css("NaiveBayesModel")

      prior = {}
      node.css("BayesOutput TargetValueCount").each do |n|
        prior[n.attribute("value").value] = n.attribute("count").value.to_f
      end

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

        # categorical
        n.css("PairCounts").each do |n2|
          boom = {}
          n2.css("TargetValueCount").each do |n3|
            boom[n3.attribute("value").value] = n3.attribute("count").value.to_f
          end
          prob[n2.attribute("value").value] = boom
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

      new(evaluator: Evaluators::NaiveBayes.new(probabilities: probabilities, features: features))
    end

    def to_pmml
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
                  v.each do |k2, v2|
                    xml.PairCounts(value: k2) do
                      xml.TargetValueCounts do
                        v2.each do |k3, v3|
                          xml.TargetValueCount(value: k3, count: v3)
                        end
                      end
                    end
                  end
                else
                  xml.TargetValueStats do
                    v.each do |k2, v2|
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
              probabilities[:prior].each do |k, v|
                xml.TargetValueCount(value: k, count: v)
              end
            end
          end
        end
      end
    end

    private

    def group_count(arr)
      r = arr.inject(Hash.new(0)) { |h, e| h[e] += 1; h }
      r.default = nil
      r
    end

    def mean(arr)
      arr.sum / arr.size.to_f
    end

    def stdev(arr)
      m = mean(arr)
      sum = arr.inject(0) { |accum, i| accum + (i - m)**2 }
      Math.sqrt(sum / (arr.length - 1).to_f)
    end
  end
end
