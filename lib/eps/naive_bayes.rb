module Eps
  class NaiveBayes < BaseEstimator
    attr_reader :probabilities

    def initialize(probabilities: nil, features: nil, target: nil)
      @probabilities = probabilities
      @features = features
      @target = target
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
      self.class.metrics(predict(@x), @y)[:accuracy]
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

      @target = node.css("BayesOutput").attribute("fieldName").value

      probabilities = {
        prior: prior,
        conditional: conditional
      }

      new(probabilities: probabilities, features: features, target: @target)
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

      Nokogiri::XML::Builder.new do |xml|
        xml.PMML(version: "4.4", xmlns: "http://www.dmg.org/PMML-4_4", "xmlns:xsi" => "http://www.w3.org/2001/XMLSchema-instance") do
          xml.Header
          pmml_data_dictionary(xml, data_fields)
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
      end.to_xml
    end

    # metrics

    def self.metrics(actual, estimated)
      Eps.metrics(actual, estimated)
    end

    private

    def _predict(x)
      probs = calculate_class_probabilities(x)
      probs.map do |xp|
        xp.sort_by { |k, v| [-v, k] }[0][0]
      end
    end

    # use log to prevent underflow
    # https://www.antoniomallia.it/lets-implement-a-gaussian-naive-bayes-classifier-in-python.html
    def calculate_class_probabilities(x)
      probs = Eps::DataFrame.new

      # assign very small probability if probability is 0
      tiny_p = 0.0001

      total = probabilities[:prior].values.sum.to_f
      probabilities[:prior].each do |c, cv|
        prior = Math.log(cv / total)
        px = [prior] * x.size

        @features.each do |k, type|
          vc = probabilities[:conditional][k][c]

          if type == "categorical"
            vc_sum = vc.values.sum

            x.columns[k].each_with_index do |xi, i|
              p2 = vc[xi].to_f / vc_sum

              # TODO use proper smoothing instead
              p2 = tiny_p if p2 == 0

              px[i] += Math.log(p2)
            end
          else
            if vc[:stdev] != 0
              x.columns[k].each_with_index do |xi, i|
                px[i] += Math.log(calculate_probability(xi, vc[:mean], vc[:stdev]))
              end
            else
              x.columns[k].each_with_index do |xi, i|
                if xi != vc[:mean]
                  # TODO use proper smoothing instead
                  px[i] += Math.log(tiny_p)
                end
              end
            end
          end

          probs.columns[c] = px
        end
      end

      probs
    end

    SQRT_2PI = Math.sqrt(2 * Math::PI)

    # TODO memoize for performance
    def calculate_probability(x, mean, stdev)
      exponent = Math.exp(-((x - mean)**2) / (2 * (stdev**2)))
      (1 / (SQRT_2PI * stdev)) * exponent
    end

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
