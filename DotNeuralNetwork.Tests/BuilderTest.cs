﻿using kDg.DotNeuralNetwork.Agents;
using kDg.DotNeuralNetwork.Middlewares;
using kDg.DotNeuralNetwork.Nets;
using TorchSharp;

namespace DotNeuralNetwork.Tests;

public class BuilderTest {
    [Test]
    public void CreateAgentTest() {
        LinearFunctionedNetBuilder builder = new("test net");
        builder.SetInputSize(2)
               .SetLayerCount(1)
               .SetPerceptronCount(4)
               .AddFunction(x => torch.nn.functional.relu(x))
               .AddFunction(x => torch.nn.functional.relu(x))
               .AddFunction(x => torch.nn.functional.sigmoid(x))
               .SetOutputSize(1);
        
        LinearFunctionedNet net = builder.Build();
        
        torch.optim.Optimizer optimizer = torch.optim.Adam(net.parameters(), 0.01);
        torch.nn.Module<torch.Tensor, torch.Tensor, torch.Tensor> lossFunction = torch.nn.MSELoss(torch.nn.Reduction.Mean);

        HistoryEpochMiddleware historyEpochMiddleware = new ();

        TrainAgentBuilder agentBuilder = new ();

        agentBuilder.SetModel(net)
                    .SetOptimizer(optimizer)
                    .SetLossFunction(lossFunction)
                    .AddEpochMiddlewares(historyEpochMiddleware);

        TrainAgent agent = agentBuilder.Build();
        
        float[,] inputData = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
        float[] outputData = { 0, 1, 1, 0 };
        
        agent.Fit(inputData, outputData, 1000);

        float[,] xData = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
        float[] predict = agent.Predict(xData);
        
        const double threshold = 0.01;
        Assert.Multiple(() => {
            Assert.That(predict[0], Is.LessThan(threshold));
            Assert.That(predict[1], Is.GreaterThan(1.0 - threshold));
            Assert.That(predict[2], Is.GreaterThan(1.0 - threshold));
            Assert.That(predict[3], Is.LessThan(threshold));
        });
    }
}
