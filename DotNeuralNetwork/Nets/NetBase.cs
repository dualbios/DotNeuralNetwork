using TorchSharp;

namespace kDg.DotNeuralNetwork.Nets;

public abstract class NetBase(string name) : torch.nn.Module<torch.Tensor, torch.Tensor>(name) {
    public abstract override torch.Tensor forward(torch.Tensor x);
}
