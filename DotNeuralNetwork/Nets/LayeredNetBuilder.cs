using System.Diagnostics;
using TorchSharp;

namespace kDg.DotNeuralNetwork.Nets;

public class LayeredNetBuilder {
    private string? _name;
    private LayeredNetInputLayer? _inputLayer;
    private readonly List<LayeredNetLayer> _layers = [];
    private LayeredNetOutputLayer? _outputLayer;

    public LayeredNetBuilder(string netName) {
        _name = netName;
    }
    
    public LayeredNetBuilder SetInputLayer(string name, int inputCount, torch.nn.Module<torch.Tensor, torch.Tensor> activation) {
        _inputLayer = new LayeredNetInputLayer(name, inputCount, activation);
        return this;
    }

    public LayeredNetBuilder AddLayer(string name, int perceptronCount, torch.nn.Module<torch.Tensor, torch.Tensor> activation) {
        _layers.Add(new LayeredNetLayer(name, perceptronCount, activation));
        return this;
    }

    public LayeredNetBuilder SetOutputLayer(string name, int outputCount, torch.nn.Module<torch.Tensor, torch.Tensor>? activation = null) {
        _outputLayer = new LayeredNetOutputLayer(name, outputCount, activation);
        return this;
    }

    public LayeredNet Build() {
        if (_inputLayer == null || _outputLayer == null || _layers.Count == 0) {
            throw new InvalidOperationException("Input layer, output layer, and at least one hidden layer must be specified.");
        }

        Debug.Assert(_name != null, nameof(_name) + " != null");
        
        return new LayeredNet(_name, _inputLayer, _layers.ToArray(), _outputLayer);
    }
}