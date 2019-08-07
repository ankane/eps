module Eps
  class DataFrame
    attr_reader :columns

    def initialize(data = [])
      @columns = {}

      if daru?(data)
        data.to_h.each do |k, v|
          @columns[k.to_s] = v.to_a
        end
      elsif data.is_a?(Hash)
        data.each do |k, v|
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

    # TODO remove
    def map
      if @columns.any?
        size.times.map do |i|
          yield Hash[@columns.map { |k, v| [k, v[i]] }]
        end
      end
    end

    def [](rows, cols = nil)
      if cols.nil?
        if rows.is_a?(String) || (rows.is_a?(Array) && rows.first.is_a?(String))
          cols = rows
          rows = 0..-1
        end
      end

      if rows.is_a?(Range)
        if rows.end.nil?
          rows = Range.new(rows.begin, size - 1)
        elsif rows.end < 0
          rows = Range.new(rows.begin, size + rows.end, rows.exclude_end?)
        end
      end

      if cols
        if cols.is_a?(Range)
          c = columns.keys

          start_index = c.index(cols.begin)
          raise "Undefined column: #{cols.begin}" unless start_index

          end_index = c.index(cols.end)
          raise "Undefined column: #{cols.end}" unless end_index

          reverse = false
          if start_index > end_index
            reverse = true
            start_index, end_index = end_index, start_index
          end

          cols = c[Range.new(start_index, end_index, cols.exclude_end?)]
          cols.reverse! if reverse
        elsif !cols.is_a?(Array)
          singular = true
          cols = [cols]
        end
      else
        cols = columns.keys
      end

      df = Eps::DataFrame.new

      cols.map(&:to_s).each do |c|
        raise "Undefined column: #{c}" unless columns.include?(c)

        df.columns[c] = columns[c].values_at(*rows)
      end

      singular ? df.columns[cols[0]] : df
    end

    def ==(other)
      columns.keys == other.columns.keys && columns == other.columns
    end

    private

    def daru?(x)
      defined?(Daru) && x.is_a?(Daru::DataFrame)
    end
  end
end
