using kDg.DotNeuralNetwork.Middlewares;

namespace CarFollower.Wpf;

public class EpochLossMiddleware : IEpochMiddleware {
    private readonly List<EpochResults> _losses = [];
    public IEnumerable<EpochResults> Losses => _losses;

    public void Reset() {
        _losses.Clear();
    }

    public void OnEpochFinished(EpochResults epochResults) {
        _losses.Add(epochResults);
    }
}
