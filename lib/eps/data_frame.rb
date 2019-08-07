module Eps
  class DataFrame
    attr_reader :columns

    def initialize(data = [])
      @columns = {}

      if daru?(data)
        data.to_h.each do |k, v|
          @columns[k.to_s] = v.to_a
        end
      else
        if data.any?
          row = data[0]

          if row.is_a?(Hash)
            row.keys.each do |k|
              @columns[k.to_s] = data.map { |r| r[k] }
            end
          elsif row.is_a?(Array)
            row.size.times do |i|
              @columns["x#{i}"] = data.map { |r| r[i] }
            end
          else
            @columns["x0"] = data
          end
        end
      end
    end

    def empty?
      @columns.empty?
    end

    def size
      @columns.any? ? columns.values.first.size : 0
    end

    def any?
      @columns.any?
    end

    def rows(idx)
      df = Eps::DataFrame.new

      self.columns.each do |k, v|
        df.columns[k] = v.values_at(*idx)
      end

      df
    end

    # TODO remove
    def map
      if @columns.any?
        size.times.map do |i|
          yield Hash[@columns.map { |k, v| [k, v[i]] }]
        end
      end
    end

    private

    def daru?(x)
      defined?(Daru) && x.is_a?(Daru::DataFrame)
    end
  end
end
