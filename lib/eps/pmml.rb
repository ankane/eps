# similar to Marshal/JSON/YAML interface
module Eps
  module PMML
    class << self
      def load(pmml)
        Loader.new(pmml).load
      end

      def generate(model)
        Generator.new(model).generate
      end
    end
  end
end
