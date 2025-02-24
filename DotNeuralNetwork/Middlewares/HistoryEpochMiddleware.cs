namespace kDg.DotNeuralNetwork.Middlewares;

public class HistoryEpochMiddleware : IEpochMiddleware {
    private List<float> Loss { get; } = new List<float>(256);

    public void Reset() {
        Loss.Clear();
    }

    public void OnEpochFinished(EpochResults epochResults) {
        Loss.Add(epochResults.Loss);
    }
}
