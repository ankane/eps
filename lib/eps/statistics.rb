module Eps
  module Statistics
    def self.normal_cdf(x, mean, std_dev)
      0.5 * (1.0 + Math.erf((x - mean) / (std_dev * Math.sqrt(2))))
    end

    # Hill, G. W. (1970).
    # Algorithm 395: Student's t-distribution.
    # Communications of the ACM, 13(10), 617-619.
    def self.students_t_cdf(x, n)
      start, sign = x < 0 ? [0, 1] : [1, -1]

      z = 1.0
      t = x * x
      y = t / n.to_f
      b = 1.0 + y

      if n > n.floor || (n >= 20.0 && t < n) || n > 200.0
        # asymptotic series for large or noninteger n
        if y > 10e-6
          y = Math.log(b)
        end
        a = n - 0.5
        b = 48.0 * a * a
        y *= a
        y = (((((-0.4 * y - 3.3) * y - 24.0) * y - 85.5) / (0.8 * y * y + 100.0 + b) + y + 3.0) / b + 1.0) * Math.sqrt(y)
        return start + sign * normal_cdf(-y, 0.0, 1.0)
      end

      if n < 20 && t < 4.0
        # nested summation of cosine series
        y = Math.sqrt(y)
        a = y
        if n == 1
          a = 0.0
        end

        # loop
        if n > 1
          n -= 2
          while n > 1
            a = (n - 1) / (b * n) * a + y
            n -= 2
          end
        end
        a = n == 0 ? a / Math.sqrt(b) : (Math.atan(y) + a / b) * (2.0 / Math::PI)
        return start + sign * (z - a) / 2.0
      end

      # tail series expanation for large t-values
      a = Math.sqrt(b)
      y = a * n
      j = 0
      while a != z
        j += 2
        z = a
        y = y * (j - 1) / (b * j)
        a += y / (n + j)
      end
      z = 0.0
      y = 0.0
      a = -a

      # loop (without n + 2 and n - 2)
      while n > 1
        a = (n - 1) / (b * n) * a + y
        n -= 2
      end
      a = n == 0 ? a / Math.sqrt(b) : (Math.atan(y) + a / b) * (2.0 / Math::PI)
      start + sign * (z - a) / 2.0
    end
  end
end
