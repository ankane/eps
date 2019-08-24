module Eps
  class Base
    class << self
      # legacy
      def build(*args)
        instance.build(*args)
      end

      def predict(*args)
        instance.predict(*args)
      end

      def train(*args)
        instance.train(*args)
      end

      private

      def instance
        @instance ||= new
      end
    end
  end
end
