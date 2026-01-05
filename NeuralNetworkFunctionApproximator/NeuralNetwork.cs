using System;
using System.Linq.Expressions;

namespace NeuralNetworkFunctionApproximator
{
    public class NeuralNetwork
    {
        private int hiddenLayerWidth;
        public int HiddenLayerWidth
        {
            get => hiddenLayerWidth;
            internal set
            {
                if (HiddenLayerWidth == value) return;

                hiddenLayerWidth = value;
                Restart();
            }
        }
        private readonly float[,] initialWeights;
        private readonly float[][] initialBiases;
        internal float[,] Weights { get; set; }
        internal float[][] Biases { get; set; }
        internal float[][] WeightedSums { get; set; }
        internal float[][] Activations { get; set; }

        private readonly Func<float, float> func;

        public NeuralNetwork(int hiddenLayerWidth, int maxHiddenLayerWidth, Func<float, float> func)
        {
            initialWeights = new float[2, maxHiddenLayerWidth];
            initialBiases = [ new float[maxHiddenLayerWidth], new float[1] ];
            Random rand = new();
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < maxHiddenLayerWidth; j++)
                    initialWeights[i, j] = (float)(rand.NextDouble() * 2 - 1);

                for (int j = 0; j < initialBiases[i].Length; j++)
                    initialBiases[i][j] = (float)(rand.NextDouble() * 2 - 1);
            }

            this.func = func;
            this.HiddenLayerWidth = hiddenLayerWidth;

            Restart();
        }

        public void Restart()
        {
            Weights = new float[2, hiddenLayerWidth];
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < hiddenLayerWidth; j++)
                    Weights[i, j] = initialWeights[i, j];
            Biases = [new float[hiddenLayerWidth], new float[1]];
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < Biases[i].Length; j++)
                    Biases[i][j] = initialBiases[i][j];

            WeightedSums = new float[2][];
            WeightedSums[0] = new float[HiddenLayerWidth];
            WeightedSums[1] = new float[1];

            Activations = new float[2][];
            Activations[0] = new float[HiddenLayerWidth];
            Activations[1] = new float[1];
        }

        public void Forward(float input)
        {
            for (int i = 0; i < HiddenLayerWidth; i++)
            {
                WeightedSums[0][i] = Weights[0, i] * input + Biases[0][i];
                Activations[0][i] = NeuralNetworkMath.Tanh(WeightedSums[0][i]);
            }

            WeightedSums[1][0] = Activations[0]
                .Select((activation, index) => activation * Weights[1, index])
                .Sum()
                + Biases[1][0];

            //Activations[1][0] = NeuralNetworkMath.Tanh(WeightedSums[1][0]);
            Activations[1][0] = WeightedSums[1][0];
        }

    }
}
