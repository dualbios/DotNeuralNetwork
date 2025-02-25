using System.Text;
using System.Windows.Controls;
using kDg.DotNeuralNetwork.Agents;
using kDg.DotNeuralNetwork.Middlewares;

namespace XorNetwork.Wpf;

public class HistoryResultsMiddleware : IEpochMiddleware {
    private TrainAgent _agent;
    public List<string> HistoryResult { get; } = new List<string>(256);
    private float[,] xData;

    public HistoryResultsMiddleware() {
        xData = new float[,]{ { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
    }

    public void Reset() {
        HistoryResult.Clear();
    }

    public void OnEpochFinished(EpochResults epochResults) {
        if (epochResults.EpochNumber % DisplayPeriod == 0) {
            float[] output = _agent.Predict(xData);
            StringBuilder stringBuilder = new();
            stringBuilder.AppendLine($"\nPrediction ({epochResults.EpochNumber}):");
            for (int i = 0; i < 4; i++) {
                stringBuilder.AppendLine($"Inputs: {xData[i, 0]}, {xData[i, 1]} → Outputs: {output[i]:F4}");
            }
            
            HistoryResult.Add(stringBuilder.ToString());
        }
    }
    
    public void SetAgent(TrainAgent agent) {
        _agent = agent;
    }
    
    public void SetDisplayPeriod(int period) {
        DisplayPeriod = period;
    }

    public int DisplayPeriod { get; private set; }
}
