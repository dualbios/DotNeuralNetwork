using System.Windows;
using kDg.DotNeuralNetwork.Agents;
using kDg.DotNeuralNetwork.Nets;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;
using TorchSharp;

namespace XorNetwork.Wpf;

/// <summary>
/// Interaction logic for MainWindow.xaml
/// </summary>
public partial class MainWindow : Window {
    private TrainAgent _agent;
    private readonly HistoryResultsMiddleware _historyResultsMiddleware;
    private readonly PlotHistoryEpochMiddleware _plotHistoryEpochMiddleware;

    public MainWindow() {
        InitializeComponent();

        PlotView.Model = new PlotModel {
            Title = "History",
            Series = { new LineSeries() },
            Axes = { new LinearAxis() { Minimum = 0 } }
        };

        _plotHistoryEpochMiddleware = new PlotHistoryEpochMiddleware(PlotView);
        _historyResultsMiddleware = new HistoryResultsMiddleware();

        _agent = CreateAgent();
    }

    private TrainAgent CreateAgent() {
        LinearFunctionedNetBuilder builder = new("test net");
        builder.SetInputSize(2)
               .SetLayerCount(1)
               .SetPerceptronCount(4)
               .AddFunction(x => torch.nn.functional.relu(x))
               //.AddFunction(x => torch.nn.functional.relu(x))
               .AddFunction(x => torch.nn.functional.sigmoid(x))
               .SetOutputSize(1);

        LinearFunctionedNet net = builder.Build();

        torch.optim.Optimizer optimizer = torch.optim.Adam(net.parameters(), 0.01);
        torch.nn.Module<torch.Tensor, torch.Tensor, torch.Tensor> lossFunction = torch.nn.MSELoss(torch.nn.Reduction.Mean);


        TrainAgentBuilder agentBuilder = new();

        agentBuilder.SetModel(net)
                    .SetOptimizer(optimizer)
                    .SetLossFunction(lossFunction)
                    .AddEpochMiddlewares(_plotHistoryEpochMiddleware)
                    .AddEpochMiddlewares(_historyResultsMiddleware);

        return agentBuilder.Build();
    }

    private void Fit_OnClick(object sender, RoutedEventArgs e) {
        float[,] inputData = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
        float[] outputData = { 0, 1, 1, 0 };

        int period = int.Parse(DisplayPeriodTextbox.Text);
        _plotHistoryEpochMiddleware.SetDisplayPeriod(period);
        _historyResultsMiddleware.SetDisplayPeriod(period);
        _historyResultsMiddleware.SetAgent(_agent);

        int epochs = int.Parse(EpochCountTextbox.Text);
        Task.Run(()=> {
            _agent.Fit(inputData, outputData, epochs);
        })
        .ContinueWith(t => {
            Dispatcher.Invoke(() => {
                ResultTextBlock.Text = string.Join(Environment.NewLine, _historyResultsMiddleware.HistoryResult);
            });
        });
    }

    private void Load_Click(object sender, RoutedEventArgs e) {
        throw new NotImplementedException();
    }

    private void ResetNet_Click(object sender, RoutedEventArgs e) {
        (PlotView.Model.Series[0] as LineSeries)!.Points.Clear();
        PlotView.InvalidatePlot(true);
        ResultTextBlock.Text = string.Empty;
        _plotHistoryEpochMiddleware.Reset();
        _historyResultsMiddleware.Reset();
        
        _agent.Dispose();
        _agent = CreateAgent();
    }

    private void Save_Click(object sender, RoutedEventArgs e) {
        throw new NotImplementedException();
    }
}
