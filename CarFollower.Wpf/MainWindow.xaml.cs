using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;

namespace CarFollower.Wpf;

/// <summary>
/// Interaction logic for MainWindow.xaml
/// </summary>
public partial class MainWindow : Window {
    private AgentEnvironment? _agentEnvironment;
    private readonly LineSeries _epochLossHistorySeries = new LineSeries();
    private EpochLossMiddleware? _epochLossMiddleware;
    private CarEnvironment? _carEnvironment;
    private const float AccelerationRate = 1f;

    public MainWindow() {
        InitializeComponent();

        PlotView.Model = new PlotModel {
            Title = "History",
            Series = { _epochLossHistorySeries },
            Axes = { new LinearAxis() { Minimum = 0 } }
        };
        
        IterationCount.Text = "200";
        
        FollowerSpeedTextbox.Text = "0";
        DistanceTextbox.Text = "30";
        LeaderSpeedTextbox.Text = "10";
    }

    private void SetupEnvironmentClick(object sender, RoutedEventArgs e) {
        _agentEnvironment?.Dispose();
        _epochLossMiddleware = new EpochLossMiddleware();
        _agentEnvironment = new AgentEnvironment(_epochLossMiddleware);
        _agentEnvironment.CreateStates((0, 100, 0.1f), (-50, 50, 1f));
        
        //FOR TEST
        string s = string.Join(Environment.NewLine, _agentEnvironment.States.Select(x=>$"{x.Distance,5} {x.Speed,5} {x.Accelerate,5} {x.None,5} {x.Break,5}"));

        //--------
        
        StatusTextBlock.Text = "Environment created.";
    }

    private void TrainClick(object sender, RoutedEventArgs e) {
        if (_agentEnvironment == null || _epochLossMiddleware == null) {
            return;
        }

        int iterationCount = int.Parse(IterationCount.Text, CultureInfo.InvariantCulture);
        
        Stopwatch sw = new();
        sw.Start();
        _agentEnvironment.Train(iterationCount);
        sw.Stop();
        
        _epochLossHistorySeries.Points.Clear();
        _epochLossHistorySeries.Points.AddRange(_epochLossMiddleware.Losses.Select(x=> new DataPoint(x.EpochNumber, x.Loss)));
        PlotView.InvalidatePlot(true);
        StatusTextBlock.Text = $"Trained in {sw.Elapsed}.";
    }

    private void RunSimulation(object sender, RoutedEventArgs e) {
        
    }

    private void StepSimulation(object sender, RoutedEventArgs e) {
        if (_agentEnvironment is null) {
            return;
        }
        
        float followerSpeed = float.Parse(FollowerSpeedTextbox.Text, CultureInfo.InvariantCulture);
        float distance = float.Parse(DistanceTextbox.Text, CultureInfo.InvariantCulture);
        float leaderSpeed = float.Parse(LeaderSpeedTextbox.Text, CultureInfo.InvariantCulture);

        _carEnvironment ??= new CarEnvironment(leaderSpeed, followerSpeed, distance);

        float[] predict = _agentEnvironment.Predict(distance, leaderSpeed - followerSpeed);
        
        int maxIndex = Array.IndexOf(predict, predict.Max());

        float acceleration = maxIndex switch {
            0 => 1f*AccelerationRate,
            2 => -1f*AccelerationRate*10f,
            _ => 0f
        };
        
        PredictedActionTextbox.Text = acceleration.ToString(CultureInfo.InvariantCulture);

        //acceleration *= AccelerationRate;

        _carEnvironment.Step(acceleration);
        
        FollowerSpeedTextbox.Text = _carEnvironment.FollowerSpeed.ToString(CultureInfo.InvariantCulture);
        DistanceTextbox.Text = _carEnvironment.Distance.ToString(CultureInfo.InvariantCulture);
        LeaderSpeedTextbox.Text = _carEnvironment.LeaderSpeed.ToString(CultureInfo.InvariantCulture);
    }

    private void ResetSimulation(object sender, RoutedEventArgs e) {
        _carEnvironment = null;
    }
}
