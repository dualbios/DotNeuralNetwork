using System.Text;
using kDg.DotNeuralNetwork.Nets;
using TorchSharp;

namespace DotNeuralNetwork.Tests;

public class Tests {
    [SetUp]
    public void Setup() {
    }

    [Test]
    public void Test1() {

        LinearFunctionedNetBuilder builder = new("test net");
        builder.SetInputSize(2)
               .SetLayerCount(1)
               .SetPerceptronCount(4)
               .AddFunction(x => torch.nn.functional.relu(x))
               .AddFunction(x => torch.nn.functional.relu(x))
               .AddFunction(x => torch.nn.functional.sigmoid(x))
               .SetOutputSize(1);
        
        LinearFunctionedNet net = builder.Build();

        float[,] xData = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
        float[] yData = { 0, 1, 1, 0 };

        torch.Tensor xTrain = torch.tensor(xData);
        torch.Tensor yTrain = torch.tensor(yData).reshape(-1, 1);

        torch.optim.Optimizer optimizer = torch.optim.Adam(net.parameters(), 0.01);
        torch.nn.Module<torch.Tensor, torch.Tensor, torch.Tensor> lossFunction = torch.nn.MSELoss(torch.nn.Reduction.Mean);

        var epochs = 1001;
        for (var epoch = 0; epoch < epochs; epoch++) {
            net.train();
            optimizer.zero_grad();

            torch.Tensor prediction = net.forward(xTrain);
            torch.Tensor loss = lossFunction.call(prediction, yTrain);

            loss.backward();
            optimizer.step();
        }

        net.eval();
        torch.Tensor output = net.forward(xTrain);
        StringBuilder stringBuilder = new();
        stringBuilder.AppendLine("\nПрогнози:");
        for (var i = 0; i < 4; i++) {
            stringBuilder.AppendLine($"Вхід: {xData[i, 0]}, {xData[i, 1]} → Вихід: {output[i].item<float>():F4}");
        }

        var result = stringBuilder.ToString();

        float[] outputArray = output.data<float>().ToArray();

        const double threshold = 0.01;
        Assert.Multiple(() => {
            Assert.That(outputArray[0], Is.LessThan(threshold));
            Assert.That(outputArray[1], Is.GreaterThan(1.0 - threshold));
            Assert.That(outputArray[2], Is.GreaterThan(1.0 - threshold));
            Assert.That(outputArray[3], Is.LessThan(threshold));
        });
    }
}
