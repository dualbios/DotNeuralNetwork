namespace kDg.DotNeuralNetwork.Exporters;

public interface IAgentExporter {
    void Export(string name, Stream stream);
}