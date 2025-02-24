using TorchSharp;

namespace kDg.DotNeuralNetwork.Nets;

public class LinearFunctionedNetBuilder {
    private readonly string _name;
    private int _inputSize;
    private int _perceptronCount;
    private int _layerCount;
    private IList<Func<torch.Tensor, torch.Tensor>> _functions = new List<Func<torch.Tensor, torch.Tensor>>();
    private int _outputSize;

    public LinearFunctionedNetBuilder(string netName) {
        _name = netName;
    }

    public LinearFunctionedNetBuilder SetInputSize(int inputSize) {
        _inputSize = inputSize;
        return this;
    }

    public LinearFunctionedNetBuilder SetPerceptronCount(int perceptronCount) {
        _perceptronCount = perceptronCount;
        return this;
    }

    public LinearFunctionedNetBuilder SetLayerCount(int layerCount) {
        _layerCount = layerCount;
        return this;
    }

    public LinearFunctionedNetBuilder AddFunction(Func<torch.Tensor, torch.Tensor> function) {
        _functions.Add(function);
        return this;
    }

    public LinearFunctionedNetBuilder SetOutputSize(int outputSize) {
        _outputSize = outputSize;
        return this;
    }

    public LinearFunctionedNet Build() {
        if (_functions.Count == 0) {
            _functions = Enumerable.Range(1, _layerCount)
                                   .Select(_ => new Func<torch.Tensor, torch.Tensor>(x => torch.nn.functional.relu(x)))
                                   .ToArray<Func<torch.Tensor, torch.Tensor>>();
        }
        return new LinearFunctionedNet(_name, _inputSize, _perceptronCount, _layerCount, _functions.ToArray(), _outputSize);
    }
}