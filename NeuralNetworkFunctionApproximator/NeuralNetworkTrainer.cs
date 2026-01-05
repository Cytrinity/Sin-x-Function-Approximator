using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkFunctionApproximator
{
    public class NeuralNetworkTrainer
    {
        public List<TrainingData> Data { get; }
        public NeuralNetwork Network { get; }
        public TrainingParameters Param { get; }

        public NeuralNetworkTrainer(TrainingParameters parameters)
        {
            Param = parameters;
            Data = [];

            Network = new(Param.HiddenNeuronCounts.Initial, Param.HiddenNeuronCounts.Final, Param.Function);
        }


        public void Train()
        {
            foreach (int neuronCount in Param.HiddenNeuronCounts.Values)
            {
                Console.WriteLine($"Starting with {neuronCount} hidden neurons...");
                Network.HiddenLayerWidth = neuronCount;

                List<float> losses = [];
                List<ApproximationPoint> results = [];
                int lossThresholdCount = 0;
                int epoch = 1;
                for (; epoch <= Param.MaxEpochs; epoch++)
                {
                    if (epoch == 1 || epoch % 1000 == 0)
                        Console.WriteLine($"Starting epoch {epoch}.");
                    float[,] weightGradients = new float[2, neuronCount];
                    float[][] biasGradients = [new float[neuronCount], new float[1]];
                    results = [];
                    
                    foreach (float input in Param.TrainingDataInputs.Values)
                        BackPropagation(input, weightGradients, biasGradients, results);

                    for (int i = 0; i < 2; i++)
                    {
                        for (int j = 0; j < Network.HiddenLayerWidth; j++)
                            Network.Weights[i, j] = Network.Weights[i, j]
                                - Param.LearningRate * (weightGradients[i, j] / Param.TrainingDataInputs.Count);

                        for (int j = 0; j < biasGradients[i].Length; j++)
                        {
                            Network.Biases[i][j] = Network.Biases[i][j]
                                - Param.LearningRate * (biasGradients[i][j] / Param.TrainingDataInputs.Count);
                        }
                    }

                    float loss = 0;
                    foreach (ApproximationPoint point in results)
                        loss += MathF.Pow(point.Output - point.Actual, 2);

                    loss /= Param.TrainingDataInputs.Count;

                    losses.Add(loss);

                    if (losses.Count >= 2 && MathF.Abs(losses[^2] - losses[^1]) < Param.LossThreshold)
                    {
                        lossThresholdCount++;
                        if (lossThresholdCount >= Param.LossThresholdLimit)
                            break;
                    }
                    else
                        lossThresholdCount = 0;
                }

                Data.Add(new(neuronCount, [.. losses], epoch, [.. results]));
            }
        }


        public void BackPropagation(float input, float[,] weightGradients, float[][] biasGradients, List<ApproximationPoint> results)
        {
            float realOutput = Param.Function(input);
            Network.Forward(input);
            results.Add(new(input, Network.Activations[1][0], realOutput));

            float deltaLoss_deltaOutputActivation = 2 * (Network.Activations[1][0] - realOutput);
            //float deltaOutputActivation_deltaOutputSum = NeuralNetworkMath.TanhDeriv(Network.WeightedSums[1][0]);
            float deltaOutputActivation_deltaOutputSum = 1;
            float deltaHiddenSum_deltaHiddenWeight = input;
            for (int i = 0; i < Network.HiddenLayerWidth; i++)
            {
                float deltaOutputSum_deltaOutputWeight = Network.Activations[0][i];
                float deltaOutputSum_deltaHiddenActivation = Network.Weights[1, i];
                float deltaHiddenActivation_deltaHiddenSum = NeuralNetworkMath.TanhDeriv(Network.WeightedSums[0][i]);

                weightGradients[1, i] += deltaLoss_deltaOutputActivation
                    * deltaOutputActivation_deltaOutputSum
                    * deltaOutputSum_deltaOutputWeight;

                weightGradients[0, i] += deltaLoss_deltaOutputActivation
                    * deltaOutputActivation_deltaOutputSum
                    * deltaOutputSum_deltaHiddenActivation
                    * deltaHiddenActivation_deltaHiddenSum
                    * deltaHiddenSum_deltaHiddenWeight;

                biasGradients[0][i] += deltaLoss_deltaOutputActivation
                    * deltaOutputActivation_deltaOutputSum
                    * deltaOutputSum_deltaHiddenActivation
                    * deltaHiddenActivation_deltaHiddenSum;
            }
            biasGradients[1][0] += deltaLoss_deltaOutputActivation
                * deltaOutputActivation_deltaOutputSum;
        }

        public void ExportDataToCsv(string filepath)
        {
            filepath = filepath.Replace(".csv", "");
            foreach (TrainingData data in Data)
            {
                using StreamWriter writer = new(filepath + data.HiddenNeuronCount + ".csv");
                writer.WriteLine("Loss,Epochs Ran,Input,Output");
                for (int i = 0; i < MathF.Max(data.EpochsRan, Param.TrainingDataInputs.Count); i++)
                {
                    string loss = data.Losses.Length > i ? data.Losses[i].ToString() : "";
                    string epochs = i == 0 ? data.EpochsRan.ToString() : "";
                    string results = data.FuncResults.Length > i ? data.FuncResults[i].Input.ToString() + "," + data.FuncResults[i].Output.ToString() : "";
                    writer.WriteLine($"{loss},{epochs},{results}");
                }
                writer.Close();
            }
        }

        public record TrainingData (int HiddenNeuronCount, float[] Losses, int EpochsRan, ApproximationPoint[] FuncResults);
        public record ApproximationPoint (float Input, float Output, float Actual);
        public record TrainingParameters (Func<float, float> Function, float LearningRate, int LossThresholdLimit, float LossThreshold, int MaxEpochs, RangeI HiddenNeuronCounts, RangeF TrainingDataInputs);
    }
}
