namespace kDg.DotNeuralNetwork.Importers;

public class FileAgentImporter : IAgentImporter {
    
    private readonly string _path;

    public FileAgentImporter(string path) {
        _path = path;
    }
    
    public void Import(string objectName, Stream stream) {
        using var fileStream = File.OpenRead(Path.Combine(_path, objectName));
        fileStream.CopyTo(stream);
    }
}
