using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkFunctionApproximator
{
    public class RangeF
    {
        public float Initial { get; }
        public float Final { get; }
        public float Step { get; }
        public int Count { get; }
        public float[] Values { get; }

        private RangeF(float initial, float final, float step, int count)
        {
            Initial = initial;
            Final = final;
            Step = step;
            Count = count;
            Values = new float[count];
            for (int i = 0; i < count; i++)
                Values[i] = initial + i * step;
        }

        public static RangeF FromCountAndFinal(float initial, float final, int count)
        {
            float step = (final - initial) / (count - 1);
            return new RangeF(initial, final, step, count);
        }

        public static RangeF FromStepAndFinal(float initial, float final, float step)
        {
            int count = (int)MathF.Round((final - initial) / (step) + 1);
            return new RangeF(initial, final, step, count);
        }

        public static RangeF FromStepAndCount(float initial, float step, int count)
        {
            float final = initial + step * (count - 1);
            return new RangeF(initial, final, step, count);
        }
    }
}
