namespace kDg.DotNeuralNetwork.Middlewares;

public interface IEpochMiddleware {
    void Reset();
    void OnEpochFinished(EpochResults epochResults);
}