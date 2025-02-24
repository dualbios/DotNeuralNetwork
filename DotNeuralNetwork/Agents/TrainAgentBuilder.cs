using kDg.DotNeuralNetwork.Middlewares;
using kDg.DotNeuralNetwork.Nets;
using TorchSharp;

namespace kDg.DotNeuralNetwork.Agents;

public class TrainAgentBuilder {
    private NetBase? _model;
    private torch.optim.Optimizer? _optimizer;
    private torch.nn.Module<torch.Tensor, torch.Tensor, torch.Tensor>? _lossFunction;
    private readonly IList<IEpochMiddleware> _epochMiddlewares = new List<IEpochMiddleware>();

    public TrainAgentBuilder SetModel(NetBase? model) {
        _model = model;
        return this;
    }

    public TrainAgentBuilder SetOptimizer(torch.optim.Optimizer? optimizer) {
        _optimizer = optimizer;
        return this;
    }

    public TrainAgentBuilder SetLossFunction(torch.nn.Module<torch.Tensor, torch.Tensor, torch.Tensor>? lossFunction) {
        _lossFunction = lossFunction;
        return this;
    }

    public TrainAgentBuilder AddEpochMiddlewares(IEpochMiddleware middleware) {
        _epochMiddlewares.Add(middleware);
        return this;
    }

    public TrainAgent Build() {
        if (_model == null) {
            throw new ArgumentNullException("Model is required");
        }
        
        if (_optimizer == null) {
            throw new ArgumentNullException("Optimizer is required");
        }
        
        if (_lossFunction == null) {
            throw new ArgumentNullException("Loss function is required");
        }
        
        return new TrainAgent(_model, _optimizer, _lossFunction, _epochMiddlewares);
    }
}
