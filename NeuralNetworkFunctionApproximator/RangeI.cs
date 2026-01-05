using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkFunctionApproximator
{
    public class RangeI
    {
        public int Initial { get; }
        public int Final { get; }
        public int Step { get; }
        public int Count { get; }
        public int[] Values { get; }

        private RangeI(int initial, int final, int step, int count)
        {
            Initial = initial;
            Final = final;
            Step = step;
            Count = count;
            Values = new int[count];
            for (int i = 0; i < count; i++)
                Values[i] = initial + i * step;
        }

        public static RangeI FromCountAndFinal(int initial, int final, int count)
        {
            int step = (final - initial) / (count - 1);
            return new RangeI(initial, final, step, count);
        }

        public static RangeI FromStepAndFinal(int initial, int final, int step)
        {
            int count = (int)MathF.Round((final - initial) / (step) + 1);
            return new RangeI(initial, final, step, count);
        }

        public static RangeI FromStepAndCount(int initial, int step, int count)
        {
            int final = initial + step * (count - 1);
            return new RangeI(initial, final, step, count);
        }
    }
}
