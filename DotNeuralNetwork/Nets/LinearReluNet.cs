using TorchSharp;
using TorchSharp.Modules;

namespace kDg.DotNeuralNetwork.Nets;

public sealed class LinearReluNet : NetBase {
    private readonly ModuleList<torch.nn.Module<torch.Tensor, torch.Tensor>> _hiddenLayers;
    private readonly torch.nn.Module<torch.Tensor, torch.Tensor> _inputLayer;
    private readonly torch.nn.Module<torch.Tensor, torch.Tensor> _outputLayer;

    public LinearReluNet(string name, int inputSize, int perceptronCount, int layerCount, int outputSize) : base(name) {
        _inputLayer = torch.nn.Linear(inputSize, perceptronCount);
        _hiddenLayers = new ModuleList<torch.nn.Module<torch.Tensor, torch.Tensor>>(Enumerable.Range(1, layerCount)
                                                                                              .Select(_ => torch.nn.Linear(perceptronCount, perceptronCount))
                                                                                              .ToArray<torch.nn.Module<torch.Tensor, torch.Tensor>>());
        _outputLayer = torch.nn.Linear(perceptronCount, outputSize);

        RegisterComponents();
    }

    protected override void Dispose(bool disposing) {
        if (disposing) {
            foreach (torch.nn.Module<torch.Tensor, torch.Tensor> module in _hiddenLayers) {
                module.Dispose();
            }

            _inputLayer.Dispose();
            _outputLayer.Dispose();
        }

        base.Dispose(disposing);
    }

    public override torch.Tensor forward(torch.Tensor x) {
        if (x.dim() == 1) {
            x = x.unsqueeze(0);
        }

        x = torch.nn.functional.relu(_inputLayer.forward(x));
        for (var index = 0; index < _hiddenLayers.Count; index++) {
            var layer = _hiddenLayers[index];
            x = torch.nn.functional.relu(layer.forward(x));
        }

        return _outputLayer.forward(x);
    }
}
