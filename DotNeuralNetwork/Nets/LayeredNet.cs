using TorchSharp;
using TorchSharp.Modules;

namespace kDg.DotNeuralNetwork.Nets;

public class LayeredNet : NetBase {
    private readonly Func<torch.Tensor, torch.Tensor>[] _functions;
    private readonly torch.nn.Module<torch.Tensor, torch.Tensor> _inputLayer;
    private readonly ModuleList<torch.nn.Module<torch.Tensor, torch.Tensor>> _hiddenLayers;
    //private readonly torch.nn.Module<torch.Tensor, torch.Tensor> _outputLayer;

    public LayeredNet(string name, LayeredNetInputLayer inputLayer, LayeredNetLayer[] layers, LayeredNetOutputLayer outputLayer) : base(name) {
        
        _inputLayer = torch.nn.Linear(inputLayer.InputCount, layers.First().PerceptronCount);
        
        _hiddenLayers =new ModuleList<torch.nn.Module<torch.Tensor, torch.Tensor>>();
        for (var index = 0; index < layers.Length-1; index++) {
            _hiddenLayers.Add(torch.nn.Linear(layers[index].PerceptronCount, layers[index+1].PerceptronCount));
        }
        
        _hiddenLayers.Add(torch.nn.Linear(layers.Last().PerceptronCount, outputLayer.OutputCount));

        _functions = new[] {inputLayer.Activation}.Concat(layers.Select(layer => layer.Activation)).ToArray();
        
        //_outputLayer = torch.nn.Linear(outputLayer.OutputCount, outputLayer.OutputCount);
    }

    public override torch.Tensor forward(torch.Tensor x)
    {
        if (x.dim() == 1) {
            x = x.unsqueeze(0);
        }

        x = _functions[0](_inputLayer.forward(x));
        for (var index = 0; index < _hiddenLayers.Count-1; index++) {
            var layer = _hiddenLayers[index];
            x = _functions[index + 1](layer.forward(x));
        }

        //return x;
        return _hiddenLayers.Last().forward(x);
    }
    
    protected override void Dispose(bool disposing) {
        if (disposing) {
            foreach (torch.nn.Module<torch.Tensor, torch.Tensor> module in _hiddenLayers) {
                module.Dispose();
            }

            _inputLayer.Dispose();
            //_outputLayer.Dispose();
        }

        base.Dispose(disposing);
    }

}

public record LayeredNetLayer {
    public string Name { get; }
    public int PerceptronCount { get; }
    public Func<torch.Tensor, torch.Tensor> Activation { get; }

    public LayeredNetLayer(string name, int perceptronCount, Func<torch.Tensor, torch.Tensor>? activation) {
        Name = name;
        PerceptronCount = perceptronCount;
        Activation = activation;
    }
}

public record LayeredNetInputLayer {
    public string Name { get; }
    public int InputCount { get; }
    public Func<torch.Tensor, torch.Tensor> Activation { get; }

    public LayeredNetInputLayer(string name, int inputCount, Func<torch.Tensor, torch.Tensor> activation) {
        Name = name;
        InputCount = inputCount;
        Activation = activation;
    }
}

public record LayeredNetOutputLayer {
    public string Name { get; }
    public int OutputCount { get; }

    public LayeredNetOutputLayer(string name, int outputCount) {
        Name = name;
        OutputCount = outputCount;
    }
}
