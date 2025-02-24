namespace kDg.DotNeuralNetwork.Nets;

public class FactoryNetProvider : INetProvider {
    private readonly Func<string, NetBase> _factory;

    public FactoryNetProvider(Func<string, NetBase> factory) {
        _factory = factory;
    }

    public NetBase Create(string name) {
        return _factory(name);
    }
}
