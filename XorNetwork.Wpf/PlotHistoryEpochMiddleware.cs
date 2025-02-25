using kDg.DotNeuralNetwork.Middlewares;
using OxyPlot;
using OxyPlot.Series;
using OxyPlot.Wpf;

namespace XorNetwork.Wpf;

public class PlotHistoryEpochMiddleware : IEpochMiddleware {
    private readonly PlotView _plotView;
    private readonly PlotModel _plotModel;
    private readonly LineSeries _lineSeries;

    public PlotHistoryEpochMiddleware(PlotView plotView) {
        _plotView = plotView;
        _plotModel = plotView.Model;
        _lineSeries = (_plotModel.Series[0] as LineSeries)!;
    }

    public void Reset() {
        (_plotModel.Series[0] as LineSeries)!.Points.Clear();
    }

    public void OnEpochFinished(EpochResults epochResults) {
        _lineSeries.Points.Add(new DataPoint(epochResults.EpochNumber, epochResults.Loss));
        if (epochResults.EpochNumber % DisplayPeriod == 0) {
            _plotView.Dispatcher.Invoke(() => _plotView.InvalidatePlot(true));
        }
    }

    public void SetDisplayPeriod(int period) {
        DisplayPeriod = period;
    }

    public int DisplayPeriod { get; private set; }
}
