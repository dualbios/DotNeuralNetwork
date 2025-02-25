namespace kDg.DotNeuralNetwork.Importers;

public interface IAgentImporter {
    void Import(string objectName, Stream stream);
}