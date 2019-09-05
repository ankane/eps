module Eps
  class Base
    class << self
      def build(*args)
        instance.build(*args)
      end

      def predict(*args)
        instance.predict(*args)
      end

      private

      def instance
        @instance ||= new
      end
    end
  end
end
