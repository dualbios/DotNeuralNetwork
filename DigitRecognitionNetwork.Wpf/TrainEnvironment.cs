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
            _agent = CreateAgent(plotHistoryEpochMiddleware);

            var random = new Random(DateTime.Now.Microsecond);
            
            float [,] inputs = new float[_imageDataList.Count, 28 * 28];
            float [,] outputs = new float[_imageDataList.Count, 10];
            
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

            // List<(float[] array1D, float[] outputs)> tuples = _imageDataList.Select(imageData => {
            //     // float[] array1D = new float[x.Image.Length];
            //     // Buffer.BlockCopy(x.Image, 0, array1D, 0, x.Image.Length * sizeof(float));
            //     
            //     float[] array1D = new float[imageData.Image.GetLength(0) * imageData.Image.GetLength(1)];
            //     for (var y = 0; y < imageData.Image.GetLength(1); y++) {
            //         for (var x = 0; x < imageData.Image.GetLength(0); x++) {
            //             array1D[x * imageData.Image.GetLength(1) + y] = imageData.Image[x, y];
            //         }
            //     }
            //
            //     float[] outputs = new float[10];
            //     outputs[imageData.Folder] = 1;
            //
            //     return (array1D, outputs);
            // }).ToList();
            //
            // float[][] floats = tuples.Select(x => x.array1D).ToArray();
            // string ggg = ToSquaredString(floats);
            // int rows = floats.Length;
            // int cols = floats[0].Length;
            //
            // float[,] inputs = new float[rows, cols];
            // for (int i = 0; i < rows; i++)
            // {
            //     for (int j = 0; j < cols; j++)
            //     {
            //         inputs[i, j] = floats[i][j];
            //     }
            // }
            // string ggg222 = ToSquaredString(inputs);
            //
            //
            // floats = tuples.Select(x => x.outputs).ToArray();
            // string fff = ToSquaredString(floats);
            //
            // rows = floats.Length;
            // cols = floats[0].Length;
            // float[,] outputs = new float[rows, cols];
            // for (int i = 0; i < rows; i++)
            // {
            //     for (int j = 0; j < cols; j++)
            //     {
            //         outputs[i, j] = floats[i][j];
            //     }
            // }
            
            _agent.Fit(inputs, outputs, 1000);

            //var imageDatas = _imageDataList.Select(x=>new {Image=x, Index=random.NextDouble()}).OrderBy(x=>x.Index).Select(x=>x.Image).ToList();
            // var iterationCount = 1000;
            // for (var i = 0; i < iterationCount; i++) {
            //     var imageData = _imageDataList[random.Next(_imageDataList.Count)];
            //     
            //     float[] outputs = new float[10];
            //     outputs[imageData.Folder] = 1;
            //
            //     // float[] array1D = new float[imageData.Image.Length];
            //     // Buffer.BlockCopy(imageData.Image, 0, array1D, 0, imageData.Image.Length * sizeof(float));
            //
            //     float[] array1D = new float[imageData.Image.GetLength(0) * imageData.Image.GetLength(1)];
            //     for (var y = 0; y < imageData.Image.GetLength(1); y++) {
            //         for (var x = 0; x < imageData.Image.GetLength(0); x++) {
            //             array1D[x * imageData.Image.GetLength(1) + y] = imageData.Image[x, y];
            //         }
            //     }
            //
            //     _agent.Fit(array1D, outputs, 100);
            //     if (i % 25 == 0) {
            //         statusCallback?.Invoke(new Status("", (double)i / iterationCount));
            //     }
            // }
        }
        catch (Exception e) {
            Console.WriteLine(e);
            throw;
        }
    }
    
    static string ToSquaredString(float[][] array)
    {
        StringBuilder sb = new StringBuilder();
        int rows = array.Length;
        int cols = array[0].Length;
        float[,] outputs = new float[rows, cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
               sb.Append($"{array[i][j].ToString("N2")} ");
            }

            sb.AppendLine();
        }

        return sb.ToString();
    }
    
    static string ToSquaredString(float[,] array)
    {
        StringBuilder sb = new StringBuilder();
        int rows = array.GetLength(0);
        int cols = array.GetLength(1);
        float[,] outputs = new float[rows, cols];
        for (int j = 0; j < cols; j++)
        {
            for (int i = 0; i < rows; i++)
            {
               sb.Append($"{array[i, j].ToString("N2")} ");
            }

            sb.AppendLine();
        }

        return sb.ToString();
    }

    private TrainAgent CreateAgent(PlotHistoryEpochMiddleware plotHistoryEpochMiddleware) {
        LinearFunctionedNetBuilder builder = new("test net");
        builder.SetInputSize(28 * 28)
               .SetLayerCount(3)
               .SetPerceptronCount(32)
               .AddFunction(x => torch.nn.functional.relu(x))
               .AddFunction(x => torch.nn.functional.relu(x))
               .AddFunction(x => torch.nn.functional.relu(x))
               .AddFunction(x => torch.nn.functional.sigmoid(x))
               //.AddFunction(x => torch.nn.functional.softmax(x, 1))
               .SetOutputSize(10);

        LinearFunctionedNet net = builder.Build();

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
