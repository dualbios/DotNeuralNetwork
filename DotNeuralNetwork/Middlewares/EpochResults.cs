namespace kDg.DotNeuralNetwork.Middlewares;

public record EpochResults(int EpochNumber, float Loss, TimeSpan ElapsedTime);
