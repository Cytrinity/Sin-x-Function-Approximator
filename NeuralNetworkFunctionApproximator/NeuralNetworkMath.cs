using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkFunctionApproximator
{
    internal static class NeuralNetworkMath
    {
        public static float Tanh(float x)
            => (MathF.Exp(2*x)-1) / (MathF.Exp(2*x)+1);

        public static float TanhDeriv(float x)
            => (4 * MathF.Exp(2 * x)) / MathF.Pow(MathF.Exp(2 * x) + 1, 2);
    }
}
