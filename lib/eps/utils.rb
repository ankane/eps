module Eps
  module Utils
    def self.column_type(c, k)
      if c.all? { |v| v.nil? }
        # goes here for empty as well
        nil
      elsif c.any? { |v| v.nil? }
        raise ArgumentError, "Nil values not allowed"
      elsif c.all? { |v| v.is_a?(Numeric) }
        "numeric"
      elsif c.all? { |v| v.is_a?(String) } # || v == true || v == false }
        "categorical"
      else
        raise ArgumentError, "Column values must be all numeric or all string: #{k}"
      end
    end
  end
end
