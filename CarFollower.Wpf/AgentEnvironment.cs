using kDg.DotNeuralNetwork.Agents;
using kDg.DotNeuralNetwork.Nets;
using TorchSharp;

namespace CarFollower.Wpf;

public class AgentEnvironment : IDisposable {
    private readonly EpochLossMiddleware _epochLossMiddleware;
    private readonly List<State> _states = [];

    private const float Threshold = 1.1f;
    private const float PerfectDistance = 30f;
    
    private TrainAgent? _agent;

    public AgentEnvironment(EpochLossMiddleware epochLossMiddleware) {
        _epochLossMiddleware = epochLossMiddleware;
    }

    public IEnumerable<State> States => _states;

    public void CreateStates((float distanceMin, float distanceMax, float step) distance,
                             (float speedMin, float speedMax, float step) speed) {
        for (float d = distance.distanceMin; d <= distance.distanceMax; d += distance.step) {
            for (float s = speed.speedMin; s <= speed.speedMax; s += speed.step) {
                (float accelerate, float none, float @break) = CreateResult(d, s);
                _states.Add(new State {
                    Distance = d,
                    Speed = s,
                    Accelerate = accelerate,
                    None = none,
                    Break = @break
                });
            }
        }
    }

    public void Train(int epochCount) {
        _agent?.Dispose();
        _agent = CreateAgent();
        
        float[,] inputs = GetInputs(_states);
        float[,] outputs = Getoutputs(_states);
        
        _agent!.Fit(inputs, outputs, epochCount);
    }

    private TrainAgent? CreateAgent() {
        NetBase net = new LayeredNetBuilder("layered test net")
                      .SetInputLayer("input", 2, torch.nn.ReLU())
                      .AddLayer("l2", 32, torch.nn.ReLU())
                      .AddLayer("l3", 32, torch.nn.ReLU())
                      .AddLayer("l4", 32, torch.nn.Sigmoid())
                      .SetOutputLayer("output", 3)
                      .Build();
        
        torch.optim.Optimizer optimizer = torch.optim.Adam(net.parameters(), 0.01);
        torch.nn.Module<torch.Tensor, torch.Tensor, torch.Tensor> lossFunction = torch.nn.MSELoss();

        TrainAgentBuilder agentBuilder = new();

        agentBuilder.SetModel(net)
                    .SetOptimizer(optimizer)
                    .SetLossFunction(lossFunction)
                    .AddEpochMiddlewares(_epochLossMiddleware);

        return agentBuilder.Build();
    }

    private (float accelerate, float none, float @break) CreateResult(float d, float s) {
        if (Math.Abs(d - PerfectDistance) < Threshold) {
            return (0, 1, 0);
        }

        return d < PerfectDistance ? (0, 0, 1) : (1, 0, 0);
    }

    private float[,] GetInputs(IList<State> states) {
        float[,] result = new float[states.Count, 2];
        for (int i = 0; i < states.Count; i++) {
            result[i, 0] = states[i].Distance;
            result[i, 1] = states[i].Speed;
        }

        return result;
    }

    private float[,] Getoutputs(IList<State> states) {
        float[,] result = new float[states.Count, 3];
        for (int i = 0; i < states.Count; i++) {
            result[i, 0] = states[i].Accelerate;
            result[i, 1] = states[i].None;
            result[i, 2] = states[i].Break;
        }

        return result;
    }

    public void Dispose() {
        // TODO release managed resources here
    }

    public float[] Predict(float distance, float followerSpeed) {
        return _agent.Predict(new[] { distance, followerSpeed });
    }
}

public class State {
    public float Distance { get; set; }
    public float Speed { get; set; }
    public float Accelerate { get; set; }
    public float None { get; set; }
    public float Break { get; set; }
}
