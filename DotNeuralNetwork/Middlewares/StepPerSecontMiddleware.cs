namespace kDg.DotNeuralNetwork.Middlewares;

public class StepPerSecontMiddleware : IEpochMiddleware {
    private List<double> StepsPerSecond { get; } = new List<double>(256);

    public void Reset() {
        StepsPerSecond.Clear();
    }

    public void OnEpochFinished(EpochResults epochResults) {
        StepsPerSecond.Add(1d / epochResults.ElapsedTime.TotalSeconds);
    }
}
