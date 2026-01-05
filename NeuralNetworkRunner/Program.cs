using NeuralNetworkFunctionApproximator;

NeuralNetworkTrainer trainer = new(new(
    (x) => MathF.Sin(x),
    0.05f,
    20,
    0.000001f,
    20_000,
    RangeI.FromStepAndFinal(1, 7, 1),
    RangeF.FromCountAndFinal(-MathF.PI, MathF.PI, 10000)));

trainer.Train();
//trainer.ExportDataToCsv("training_data_");