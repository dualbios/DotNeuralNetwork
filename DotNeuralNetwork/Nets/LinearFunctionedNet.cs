using TorchSharp;
using TorchSharp.Modules;

namespace kDg.DotNeuralNetwork.Nets;

public sealed class LinearFunctionedNet : NetBase {
    private readonly Func<torch.Tensor, torch.Tensor>[] _functions;
    private readonly torch.nn.Module<torch.Tensor, torch.Tensor> _inputLayer;
    private readonly ModuleList<torch.nn.Module<torch.Tensor, torch.Tensor>> _hiddenLayers;
    private readonly torch.nn.Module<torch.Tensor, torch.Tensor> _outputLayer;

    public LinearFunctionedNet(string name, int inputSize, int perceptronCount, int layerCount, Func<torch.Tensor, torch.Tensor>[] functions, int outputSize) : base(name) {
        if (functions.Length != layerCount + 1) {
            throw new ArgumentException("The number of functions must be equal to the number of layers plus 1 for input layer.");
        }

        _functions = functions;
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

        x = _functions[0](_inputLayer.forward(x));
        for (var index = 0; index < _hiddenLayers.Count; index++) {
            var layer = _hiddenLayers[index];
            x = _functions[index + 1](layer.forward(x));
        }

        //return _functions[^1](_outputLayer.forward(x));
        return _outputLayer.forward(x);
    }
}
