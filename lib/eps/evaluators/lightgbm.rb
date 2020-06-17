module Eps
  module Evaluators
    class LightGBM
      attr_reader :features

      def initialize(trees:, objective:, labels:, features:, text_features:)
        @trees = trees
        @objective = objective
        @labels = labels
        @features = features
        @text_features = text_features
      end

      def predict(data, probabilities: false)
        raise "Probabilities not supported" if probabilities && @objective == "regression"

        rows = data.map(&:to_h)

        # sparse matrix
        @text_features.each do |k, v|
          encoder = TextEncoder.new(**v)
          counts = encoder.transform(data.columns[k])

          counts.each_with_index do |xc, i|
            row = rows[i]
            row.delete(k)
            xc.each do |word, count|
              row[[k, word]] = count
            end
          end
        end

        case @objective
        when "regression"
          sum_trees(rows, @trees)
        when "binary"
          prob = sum_trees(rows, @trees).map { |s| sigmoid(s) }
          if probabilities
            prob.map { |v| @labels.zip([1 - v, v]).to_h }
          else
            prob.map { |v| @labels[v > 0.5 ? 1 : 0] }
          end
        else
          tree_scores = []
          num_trees = @trees.size / @labels.size
          @trees.each_slice(num_trees).each do |trees|
            tree_scores << sum_trees(rows, trees)
          end
          rows.size.times.map do |i|
            v = tree_scores.map { |s| s[i] }
            if probabilities
              exp = v.map { |vi| Math.exp(vi) }
              sum = exp.sum
              @labels.zip(exp.map { |e| e / sum }).to_h
            else
              idx = v.map.with_index.max_by { |v2, _| v2 }.last
              @labels[idx]
            end
          end
        end
      end

      private

      def sum_trees(data, trees)
        data.map do |row|
          sum = 0
          trees.each do |node|
            score = node_score(node, row)
            sum += score
          end
          sum
        end
      end

      def matches?(node, row)
        if node.predicate.nil?
          true
        else
          v = row[node.field]

          # sparse text feature
          v = 0 if v.nil? && node.field.is_a?(Array)

          if v.nil?
            # missingValueStrategy="none"
            false
          else
            case node.operator
            when "equal"
              v.to_s == node.value
            when "in"
              node.value.include?(v)
            when "greaterThan"
              v > node.value
            when "lessOrEqual"
              v <= node.value
            else
              raise "Unknown operator: #{node.operator}"
            end
          end
        end
      end

      def node_score(node, row)
        if matches?(node, row)
          node.children.each do |c|
            score = node_score(c, row)
            return score if score
          end
          # noTrueChildStrategy="returnLastPrediction"
          node.score
        else
          nil
        end
      end

      def sigmoid(x)
        1.0 / (1 + Math.exp(-x))
      end
    end
  end
end
