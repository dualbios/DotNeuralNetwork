using System.Diagnostics;
using System.IO;
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
using Microsoft.Win32;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using Image = SixLabors.ImageSharp.Image;

namespace DigitRecognitionNetwork.Wpf;

/// <summary>
/// Interaction logic for MainWindow.xaml
/// </summary>
public partial class MainWindow : Window {
    private readonly IDictionary<string, float[,]> _images = new Dictionary<string, float[,]>();
    private TrainEnvironment _trainEnvironment;
    private PlotHistoryEpochMiddleware _plotHistoryEpochMiddleware;
    readonly LineSeries _barSeries = new LineSeries() { Color = OxyColors.Blue};

    public MainWindow() {
        InitializeComponent();

        PlotView.Model = new PlotModel {
            Title = "History",
            Series = { new LineSeries() },
            Axes = { new LinearAxis() { Minimum = 0 } }
        };

        
        ResultPlotView.Model = new PlotModel {
            Title = "Result",
            Series = { _barSeries },
            Axes = { new LinearAxis() { Minimum = 0, Maximum = 1} }
        };

        _plotHistoryEpochMiddleware = new PlotHistoryEpochMiddleware(PlotView);
    }

    private void ReadFile_Click(object sender, RoutedEventArgs e) {
        OpenFolderDialog folderDialog = new() {
            Title = "Select the folder to import"
        };

        if (folderDialog.ShowDialog() == true) {
            _trainEnvironment = new TrainEnvironment(folderDialog.FolderName);
            Task.Run(async () => await _trainEnvironment.ReadImagesAsync(StatusCallback));
        }
    }

    private void StatusCallback(Status status) {
        Dispatcher.Invoke(() => {
            if (status.Progress * 100 % 25 == 0) {
                StatusTextBlock.Text = status.Progress.ToString("P2");
                (PlotView.Model.Series[0] as LineSeries)!.Points.Clear();
            }
        });
    }


    private float[,] ReadImage(string imagePath) {
        using Image<Rgba32> image = Image.Load<Rgba32>(imagePath);
        var result = new float[image.Height, image.Width];
        for (var y = 0; y < image.Height; y++) {
            for (var x = 0; x < image.Width; x++) {
                Rgba32 pixel = image[x, y];
                result[x, y] = 0.3f * pixel.R + 0.59f * pixel.G + 0.11f * pixel.B;
            }
        }

        return result;
    }

    private void Train_Click(object sender, RoutedEventArgs e) {
        _plotHistoryEpochMiddleware = new PlotHistoryEpochMiddleware(PlotView);
        (PlotView.Model.Series[0] as LineSeries)!.Points.Clear();
        PlotView.Model.InvalidatePlot(true);
        _barSeries.Points.Clear();
        ResultPlotView.Model.InvalidatePlot(true);

        Task.Run(async () => await _trainEnvironment.TrainAsync(_plotHistoryEpochMiddleware, TrainStatusCallback))
            .ContinueWith(t => Dispatcher.Invoke(() => {
                StatusTextBlock.Text = "Done!";
            }));
    }

    private void TrainStatusCallback(Status status) {
        Dispatcher.Invoke(() => {
            StatusTextBlock.Text = status.Progress.ToString("P2");
            (PlotView.Model.Series[0] as LineSeries)!.Points.Clear();
            PlotView.Model.InvalidatePlot(true);
        });
    }

    private void Predict_Click(object sender, RoutedEventArgs e) {
        bool[,] values = DrawingControl.GetPixelValues();

        float[] pixelValues = new float[values.GetLength(0) * values.GetLength(1)];
        for (var y = 0; y < values.GetLength(1); y++) {
            for (var x = 0; x < values.GetLength(0); x++) {
                pixelValues[x * values.GetLength(1) + y] = values[x, y] ? 1f : 0f;
            }
        }

        float[] pred = _trainEnvironment.Predict(pixelValues);
        float maxPrediction = pred.Max();
        int maxIndex = Array.IndexOf(pred, maxPrediction);
        StatusTextBlock.Text = $"Predicted: {maxIndex}.";
        
        _barSeries.Points.Clear();
        for (int i = 0; i < pred.Length; i++) {
            _barSeries.Points.Add(new DataPoint(i, pred[i]) );
        }
        ResultPlotView.Model.InvalidatePlot(true);
    }

    private void Clear_Click(object sender, RoutedEventArgs e) {
        DrawingControl.Clear();
    }
}
