module Eps
  class DataFrame
    attr_reader :columns
    attr_accessor :label, :weight

    def initialize(data = [])
      @columns = {}

      if data.is_a?(Eps::DataFrame)
        data.columns.each do |k, v|
          @columns[k] = v
        end
      elsif rover?(data) || daru?(data)
        data.to_h.each do |k, v|
          @columns[k.to_s] = v.to_a
        end
      elsif data.is_a?(Hash)
        data.each do |k, v|
          @columns[k.to_s] = v.to_a
        end
      else
        data = data.to_a if numo?(data)

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
      size == 0
    end

    def size
      @columns.any? ? columns.values.first.size : 0
    end

    def any?
      @columns.any?
    end

    def map
      if @columns.any?
        size.times.map do |i|
          yield @columns.to_h { |k, v| [k, v[i]] }
        end
      end
    end

    def map_rows
      if @columns.any?
        size.times.map do |i|
          yield @columns.map { |_, v| v[i] }
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
        else
          finish = rows.end
          finish -= 1 if rows.exclude_end?
          rows = Range.new(rows.begin, size - 1) if finish >= size - 1
        end
      elsif rows.is_a?(Integer)
        rows = [rows]
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

      cols.each do |c|
        raise "Undefined column: #{c}" unless columns.include?(c)

        col = columns[c]
        df.columns[c] = rows.map { |i| col[i] }
      end
      df.label = rows.map { |i| label[i] } if label
      df.weight = rows.map { |i| weight[i] } if weight

      singular ? df.columns[cols[0]] : df
    end

    def ==(other)
      columns.keys == other.columns.keys && columns == other.columns
    end

    def dup
      df = Eps::DataFrame.new
      columns.each do |k, v|
        df.columns[k] = v
      end
      df.label = label
      df.weight = weight
      df
    end

    private

    def numo?(x)
      defined?(Numo::NArray) && x.is_a?(Numo::NArray)
    end

    def rover?(x)
      defined?(Rover::DataFrame) && x.is_a?(Rover::DataFrame)
    end

    def daru?(x)
      defined?(Daru::DataFrame) && x.is_a?(Daru::DataFrame)
    end
  end
end
