module Eps
  class Base
    class << self
      def build
        instance.build
      end

      def predict
        instance.predict
      end

      private

      def instance
        @instance ||= new
      end
    end
  end
end
