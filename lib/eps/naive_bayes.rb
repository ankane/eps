module Eps
  class NaiveBayes < BaseEstimator
    attr_reader :probabilities

    def initialize(probabilities: nil, target: nil)
      @probabilities = probabilities if probabilities
      @target = target if target
    end

    def train(*args)
      super

      @y = @y.map { |yi| yi.to_s }

      prior = group_count(@y)
      conditional = {}

      if @x.any?
        keys = @x.first.keys
        x = @x.dup
        x.each_with_index do |xi, i|
          xi[@target] = @y[i]
        end
        keys.each do |k|
          conditional[k] = {}
          x.group_by { |xi| xi[@target] }.each do |group, xs|
            v = xs.map { |xi| xi[k] }

            if categorical?(v[0])
              # TODO apply smoothing
              # apply smoothing only to
              # 1. categorical features
              # 2. conditional probabilities
              # TODO more efficient count
              conditional[k][group] = group_count(v)
            else
              conditional[k][group] = {mean: mean(v), stdev: stdev(v)}
            end
          end
        end
      end

      @probabilities = {
        prior: prior,
        conditional: conditional
      }
    end

    # TODO better summary
    def summary(extended: false)
      @summary_str ||= begin
        str = String.new("")
        probabilities[:prior].each do |k, v|
          str += "#{k}: #{v}\n"
        end
        str += "\n"
        str += "accuracy: %d%%\n" % [(100 * accuracy).round]
        str
      end
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
      node.css("BayesInput").each do |n|
        prob = {}
        n.css("TargetValueStat").each do |n2|
          n3 = n2.css("GaussianDistribution")
          prob[n2.attribute("value").value] = {
            mean: n3.attribute("mean").value.to_f,
            stdev: Math.sqrt(n3.attribute("variance").value.to_f)
          }
        end
        n.css("PairCounts").each do |n2|
          boom = {}
          n2.css("TargetValueCount").each do |n3|
            boom[n3.attribute("value").value] = n3.attribute("count").value.to_f
          end
          prob[n2.attribute("value").value] = boom
        end
        conditional[n.attribute("fieldName").value] = prob
      end

      @target = node.css("BayesOutput").attribute("fieldName").value

      probabilities = {
        prior: prior,
        conditional: conditional
      }

      new(probabilities: probabilities, target: @target)
    end

    def to_pmml
      data_fields = {}
      data_fields[@target] = probabilities[:prior].keys
      probabilities[:conditional].each do |k, v|
        if !v.values[0][:mean]
          data_fields[k] = v.keys
        else
          data_fields[k] = nil
        end
      end

      builder = Nokogiri::XML::Builder.new do |xml|
        xml.PMML(version: "4.3", xmlns: "http://www.dmg.org/PMML-4_3", "xmlns:xsi" => "http://www.w3.org/2001/XMLSchema-instance") do
          xml.Header
          xml.DataDictionary do
            data_fields.each do |k, vs|
              if vs
                xml.DataField(name: k, optype: "categorical", dataType: "string") do
                  vs.each do |v|
                    xml.Value(value: v)
                  end
                end
              else
                xml.DataField(name: k, optype: "continuous", dataType: "double")
              end
            end
          end
          xml.NaiveBayesModel(functionName: "classification", threshold: 0.001) do
            xml.MiningSchema do
              data_fields.each do |k, _|
                xml.MiningField(name: k)
              end
            end
            xml.BayesInputs do
              probabilities[:conditional].each do |k, v|
                xml.BayesInput(fieldName: k) do
                  if !v.values[0][:mean]
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
      {
        accuracy: actual.zip(estimated).count { |yi, yi2| yi == yi2 } / actual.size.to_f
      }
    end

    private

    def _predict(x)
      x.map do |xi|
        probs = calculate_class_probabilities(xi)
        # deterministic for equal probabilities
        probs.sort_by { |k, v| [-v, k.to_s] }[0][0]
      end
    end

    def calculate_class_probabilities(x)
      prob = {}
      probabilities[:prior].each do |c, cv|
        prob[c] = cv.to_f / probabilities[:prior].values.sum
        probabilities[:conditional].each do |k, v|
          if !v[c][:mean]
            # TODO compute ahead of time
            p2 = v[c][x[k]].to_f / v[c].values.sum

            # assign very small probability if probability is 0
            # TODO use proper smoothing instead
            if p2 == 0
              p2 = 0.0001
            end

            prob[c] *= p2
          else
            prob[c] *= calculate_probability(x[k], v[c][:mean], v[c][:stdev])
          end
        end
      end
      prob
    end

    def calculate_probability(x, mean, stdev)
      exponent = Math.exp(-((x - mean)**2) / (2 * (stdev**2)))
      (1 / (Math.sqrt(2 * Math::PI) * stdev)) * exponent
    end

    def group_count(arr)
      r = arr.inject(Hash.new(0)) { |h, e| h[e.to_s] += 1 ; h }
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
