namespace kDg.DotNeuralNetwork.Exporters;

public class FileAgentExporter : IAgentExporter {
    private readonly string _path;

    public FileAgentExporter(string path) {
        _path = path;
    }

    public void Export(string name, Stream stream) {
        using var fileStream = File.Create(Path.Combine(_path, name));
        stream.CopyTo(fileStream);
    }
}
