using TorchSharp;

namespace kDg.DotNeuralNetwork.Nets;

public sealed class LayeredNet : NetBase {
    private torch.nn.Module<torch.Tensor, torch.Tensor> _model = null;

    public LayeredNet(string name, LayeredNetInputLayer inputLayer, LayeredNetLayer[] layers, LayeredNetOutputLayer outputLayer) : base(name) {
        var l = new List<Tuple<string, torch.nn.Module<torch.Tensor, torch.Tensor>>> {
            new(inputLayer.Name, torch.nn.Linear(inputLayer.InputCount, layers.First().PerceptronCount)),
            new(inputLayer.Name + "activation", inputLayer.Activation)
        };

        for (var index = 0; index < layers.Length - 1; index++) {
            LayeredNetLayer layer = layers[index];
            l.Add(new Tuple<string, torch.nn.Module<torch.Tensor, torch.Tensor>>(layer.Name, torch.nn.Linear(layer.PerceptronCount, layers[index + 1].PerceptronCount)));
            l.Add(new Tuple<string, torch.nn.Module<torch.Tensor, torch.Tensor>>(layer.Name + "activation", layer.Activation));
        }

        l.Add(new Tuple<string, torch.nn.Module<torch.Tensor, torch.Tensor>>(inputLayer.Name, torch.nn.Linear(layers.Last().PerceptronCount, outputLayer.OutputCount)));
        if (outputLayer.Activation is not null) {
            l.Add(new Tuple<string, torch.nn.Module<torch.Tensor, torch.Tensor>>(outputLayer.Name + "activation", outputLayer.Activation));
        }

        _model = torch.nn.Sequential(l);

        RegisterComponents();
    }

    protected override void Dispose(bool disposing) {
        _model?.Dispose();

        base.Dispose(disposing);
    }

    public override torch.Tensor forward(torch.Tensor x) {
        return _model.forward(x);
    }
}

public record LayeredNetLayer(string Name, int PerceptronCount, torch.nn.Module<torch.Tensor, torch.Tensor> Activation);

public record LayeredNetInputLayer(string Name, int InputCount, torch.nn.Module<torch.Tensor, torch.Tensor> Activation);

public record LayeredNetOutputLayer(string Name, int OutputCount, torch.nn.Module<torch.Tensor, torch.Tensor>? Activation = null);
