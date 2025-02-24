namespace kDg.DotNeuralNetwork.Nets;

public interface INetProvider {
    NetBase Create(string name);
}
