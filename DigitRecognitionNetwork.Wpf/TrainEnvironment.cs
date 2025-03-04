using System.IO;
using System.Text;
using kDg.DotNeuralNetwork.Agents;
using kDg.DotNeuralNetwork.Nets;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using TorchSharp;

namespace DigitRecognitionNetwork.Wpf;

public class TrainEnvironment {
    public string TrainSourcePath { get; }

    private readonly List<ImageData> _imageDataList = new();
    private TrainAgent _agent;

    public TrainEnvironment(string trainSourcePath) {
        TrainSourcePath = trainSourcePath;
    }

    public async Task ReadImagesAsync(Action<Status> statusCallback) {
        _imageDataList.Clear();
        IEnumerable<string> files = Enumerable.Range(0, 10)
                                              .Select(i => Path.Combine(TrainSourcePath, i.ToString()))
                                              .SelectMany(Directory.GetFiles)
                                              .ToList();

        foreach (string file in files.AsParallel()) {
            string lastFolder = Path.GetFileName(Path.GetDirectoryName(file)) ?? string.Empty;
            int lastFolderIndex = int.Parse(lastFolder);
            ImageData imageData = new(ReadImage(file), lastFolderIndex);
            _imageDataList.Add(imageData);
            statusCallback?.Invoke(new Status("", (double)_imageDataList.Count / files.Count()));
        }
    }

    private float[,] ReadImage(string imagePath) {
        using Image<Rgba32> image = Image.Load<Rgba32>(imagePath);
        var result = new float[image.Height, image.Width];
        for (var y = 0; y < image.Height; y++) {
            for (var x = 0; x < image.Width; x++) {
                Rgba32 pixel = image[x, y];
                result[x, y] = (0.3f * pixel.R + 0.59f * pixel.G + 0.11f * pixel.B) / 255f;
            }
        }

        return result;
    }

    public async Task TrainAsync(PlotHistoryEpochMiddleware plotHistoryEpochMiddleware, Action<Status> statusCallback) {
        try {
            _agent?.Dispose();
            _agent = CreateAgent(plotHistoryEpochMiddleware);

            var random = new Random(DateTime.Now.Microsecond);

            float[,] inputs = new float[_imageDataList.Count, 28 * 28];
            float[,] outputs = new float[_imageDataList.Count, 10];

            for (var i = 0; i < _imageDataList.Count; i++) {
                float[,] image = _imageDataList[i].Image;
                for (var y = 0; y < image.GetLength(1); y++) {
                    for (var x = 0; x < image.GetLength(0); x++) {
                        inputs[i, x * image.GetLength(1) + y] = image[x, y];
                    }
                }
            }

            for (var i = 0; i < _imageDataList.Count; i++) {
                outputs[i, _imageDataList[i].Folder] = 1;
            }

            _agent.Fit(inputs, outputs, 1000);
        }
        catch (Exception e) {
            Console.WriteLine(e);
            throw;
        }
    }

    static string ToSquaredString(float[][] array) {
        StringBuilder sb = new StringBuilder();
        int rows = array.Length;
        int cols = array[0].Length;
        float[,] outputs = new float[rows, cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                sb.Append($"{array[i][j].ToString("N2")} ");
            }

            sb.AppendLine();
        }

        return sb.ToString();
    }

    static string ToSquaredString(float[,] array) {
        StringBuilder sb = new StringBuilder();
        int rows = array.GetLength(0);
        int cols = array.GetLength(1);
        float[,] outputs = new float[rows, cols];
        for (int j = 0; j < cols; j++) {
            for (int i = 0; i < rows; i++) {
                sb.Append($"{array[i, j].ToString("N2")} ");
            }

            sb.AppendLine();
        }

        return sb.ToString();
    }

    private TrainAgent CreateAgent(PlotHistoryEpochMiddleware plotHistoryEpochMiddleware) {
        NetBase net = new LayeredNetBuilder("layered test net")
            .SetInputLayer("input", 28 * 28, torch.nn.ReLU())
            .AddLayer("l2", 32, torch.nn.ReLU())
            .AddLayer("l3", 32, torch.nn.ReLU())
            .AddLayer("l4", 32, torch.nn.Sigmoid())
            .SetOutputLayer("output", 10)
            .Build();
        
        torch.optim.Optimizer optimizer = torch.optim.Adam(net.parameters(), 0.01);
        //torch.nn.Module<torch.Tensor, torch.Tensor, torch.Tensor> lossFunction = torch.nn.MSELoss();
        torch.nn.Module<torch.Tensor, torch.Tensor, torch.Tensor> lossFunction = torch.nn.CrossEntropyLoss();

        TrainAgentBuilder agentBuilder = new();

        agentBuilder.SetModel(net)
                    .SetOptimizer(optimizer)
                    .SetLossFunction(lossFunction)
                    .AddEpochMiddlewares(plotHistoryEpochMiddleware);

        return agentBuilder.Build();
    }

    public float[] Predict(float[] pixelValues) {
        return _agent.Predict(pixelValues);
    }
}

public record Status(string Message, double Progress);

public record ImageData(float[,] Image, int Folder);
