module Eps
  module Evaluators
    class Node
      attr_accessor :score, :predicate, :children, :leaf_index

      def initialize(predicate: nil, score: nil, children: nil, leaf_index: nil)
        @predicate = predicate
        @children = children || []
        @score = score
        @leaf_index = leaf_index
      end

      def field
        @predicate[:field]
      end

      def operator
        @predicate[:operator]
      end

      def value
        @predicate[:value]
      end
    end
  end
end
