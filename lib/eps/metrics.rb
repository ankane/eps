module Eps
  module Metrics
    class << self
      def rmse(y_true, y_pred, weight: nil)
        check_size(y_true, y_pred)
        Math.sqrt(mean(errors(y_true, y_pred).map { |v| v**2 }, weight: weight))
      end

      def mae(y_true, y_pred, weight: nil)
        check_size(y_true, y_pred)
        mean(errors(y_true, y_pred).map { |v| v.abs }, weight: weight)
      end

      def me(y_true, y_pred, weight: nil)
        check_size(y_true, y_pred)
        mean(errors(y_true, y_pred), weight: weight)
      end

      def accuracy(y_true, y_pred, weight: nil)
        check_size(y_true, y_pred)
        values = y_true.zip(y_pred).map { |yt, yp| yt == yp ? 1 : 0 }
        if weight
          values.each_with_index do |v, i|
            values[i] *= weight[i]
          end
          values.sum / weight.sum.to_f
        else
          values.sum / y_true.size.to_f
        end
      end

      # http://wiki.fast.ai/index.php/Log_Loss
      def log_loss(y_true, y_pred, eps: 1e-15, weight: nil)
        check_size(y_true, y_pred)
        p = y_pred.map { |yp| yp.clamp(eps, 1 - eps) }
        mean(y_true.zip(p).map { |yt, pi| yt == 1 ? -Math.log(pi) : -Math.log(1 - pi) }, weight: weight)
      end

      private

      def check_size(y_true, y_pred)
        raise ArgumentError, "Different sizes" if y_true.size != y_pred.size
      end

      def mean(arr, weight: nil)
        if weight
          arr.map.with_index { |v, i| v * weight[i] }.sum / weight.sum.to_f
        else
          arr.sum / arr.size.to_f
        end
      end

      def errors(y_true, y_pred)
        y_true.zip(y_pred).map { |yt, yp| yt - yp }
      end
    end
  end
end
