using System.Diagnostics;
using kDg.DotNeuralNetwork.Exporters;
using kDg.DotNeuralNetwork.Importers;
using kDg.DotNeuralNetwork.Middlewares;
using kDg.DotNeuralNetwork.Nets;
using TorchSharp;

namespace kDg.DotNeuralNetwork.Agents;

public class TrainAgent : IDisposable {
    private readonly IEnumerable<IEpochMiddleware> _epochMiddlewares = [];

    public TrainAgent(NetBase model,
                      torch.optim.Optimizer optimizer,
                      torch.nn.Module<torch.Tensor, torch.Tensor, torch.Tensor> lossFunction) {
        Model = model;
        Optimizer = optimizer;
        LossFunction = lossFunction;
    }

    public TrainAgent(NetBase model,
                      torch.optim.Optimizer optimizer, 
                      torch.nn.Module<torch.Tensor, torch.Tensor, torch.Tensor> lossFunction,
                      IEnumerable<IEpochMiddleware> epochMiddlewares)
        : this(model, optimizer, lossFunction) {
        _epochMiddlewares = epochMiddlewares;
    }

    public torch.nn.Module<torch.Tensor, torch.Tensor, torch.Tensor> LossFunction { get; }
    public NetBase Model { get; }
    public torch.optim.Optimizer Optimizer { get; }

    public void Fit(float[,] inputData, float[] outputData, int epochs) {
        torch.Tensor input = torch.tensor(inputData);
        torch.Tensor output = torch.tensor(outputData);
        Fit(input, output, epochs);
    }

    public void Fit(float[,] inputData, float[,] outputData, int epochs) {
        torch.Tensor input = torch.tensor(inputData);
        torch.Tensor output = torch.tensor(outputData);
        Fit(input, output, epochs);
    }

    public void Fit(float[] inputData, float[,] outputData, int epochs) {
        torch.Tensor input = torch.tensor(inputData);
        torch.Tensor output = torch.tensor(outputData);
        Fit(input, output, epochs);
    }


    public void Fit(float[] inputData, float[] outputData, int epochs) {
        torch.Tensor input = torch.tensor(inputData);
        torch.Tensor output = torch.tensor(outputData).reshape(1, -1);
        Fit(input, output, epochs);
    }

    private void Fit(torch.Tensor input, torch.Tensor output, int epochs) {
        if (Model == null) {
            throw new ArgumentNullException(nameof(Model));
        }

        if (Optimizer == null) {
            throw new ArgumentNullException(nameof(Optimizer));
        }
        
        if (LossFunction == null) {
            throw new ArgumentNullException(nameof(LossFunction));
        }
        
        
        Stopwatch stopwatch = new();
        stopwatch.Reset();
        for (var epoch = 0; epoch < epochs; epoch++) {
            stopwatch.Restart();
            Model.train();
            Optimizer.zero_grad();

            torch.Tensor prediction = Model.forward(input);
            torch.Tensor loss = LossFunction.call(prediction, output);

            loss.backward();
            Optimizer.step();

            stopwatch.Stop();

            var lossItem = loss.item<float>();

            foreach (IEpochMiddleware epochMiddleware in _epochMiddlewares) {
                epochMiddleware.OnEpochFinished(new EpochResults(epoch, lossItem, stopwatch.Elapsed));
            }
        }
    }


    public float[] Predict(float[] input) {
        torch.Tensor inputTensor = torch.tensor(input);
        return Predict(inputTensor).data<float>().ToArray();
    }

    public float[] Predict(float[,] input) {
        torch.Tensor inputTensor = torch.tensor(input);
        return Predict(inputTensor).data<float>().ToArray();
    }

    private torch.Tensor Predict(torch.Tensor input) {
        if (Model == null) {
            throw new ArgumentNullException(nameof(Model));
        }
        return Model.forward(input);
    }

    public float[,] Predict2D(float[] input) {
        torch.Tensor inputTensor = torch.tensor(input);
        return Predict2D(inputTensor);
    }

    public float[,] Predict2D(float[,] input) {
        torch.Tensor inputTensor = torch.tensor(input);
        return Predict2D(inputTensor);
    }

    private float[,] Predict2D(torch.Tensor inputTensor) {
        torch.Tensor tensor = Predict(inputTensor).data<float>().ToArray();

        float[] data = tensor.data<float>().ToArray();

        long[] shape = tensor.shape;
        var rows = (int)shape[0];
        var cols = (int)shape[1];

        var array = new float[rows, cols];

        for (var i = 0; i < rows; i++)
        for (var j = 0; j < cols; j++) {
            array[i, j] = data[i * cols + j];
        }

        return array;
    }

    public void Dispose() {
        LossFunction.Dispose();
        Model.Dispose();
        Optimizer.Dispose();
    }

    public void Export(IAgentExporter exporter) {
        MemoryStream memoryStream = new();
        BinaryWriter streamWriter = new(memoryStream);
        Model.save(streamWriter);
        
        memoryStream.Seek(0, SeekOrigin.Begin);
        exporter.Export("model", memoryStream);
    }

    public void Import(IAgentImporter importer) {
        Stream stream = new MemoryStream();
        importer.Import("model", stream);
        stream.Seek(0, SeekOrigin.Begin);
        Model.load(stream);
    }
}
