using System.Windows;

namespace DigitRecognitionNetwork.Wpf;

using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Shapes;

public class DrawingControl : Canvas {
    private const int Size = 28;
    private const double CellSize = 10;
    private readonly Rectangle[,] _pixels;
    private readonly SolidColorBrush _drawColor = Brushes.Black;
    private readonly SolidColorBrush _cleanColor = Brushes.White;

    public static readonly DependencyProperty PenSizeProperty = DependencyProperty.Register(nameof(PenSize),
                                                                                            typeof(int),
                                                                                            typeof(DrawingControl),
                                                                                            new PropertyMetadata(1));

    public int PenSize {
        get => (int)GetValue(PenSizeProperty);
        set => SetValue(PenSizeProperty, value);
    }

    public DrawingControl() {
        Width = Size * CellSize;
        Height = Size * CellSize;
        Background = Brushes.White;
        _pixels = new Rectangle[Size, Size];
        InitializeGrid();
        MouseMove += OnMouseMove;
        MouseDown += OnMouseMove;
    }


    private void InitializeGrid() {
        for (var x = 0; x < Size; x++) {
            for (var y = 0; y < Size; y++) {
                var rect = new Rectangle {
                    Width = CellSize,
                    Height = CellSize,
                    Fill = Brushes.White,
                    Stroke = Brushes.LightGray,
                    StrokeThickness = 0.5
                };
                SetLeft(rect, x * CellSize);
                SetTop(rect, y * CellSize);
                Children.Add(rect);
                _pixels[x, y] = rect;
            }
        }
    }

    public bool[,] GetPixelValues() {
        var pixelValues = new bool[Size, Size];
        for (var x = 0; x < Size; x++) {
            for (var y = 0; y < Size; y++) {
                pixelValues[x, y] = _pixels[x, y].Fill == _drawColor;
            }
        }

        return pixelValues;
    }

    private void OnMouseMove(object sender, MouseEventArgs e) {
        if (e.LeftButton == MouseButtonState.Pressed) {
            Point position = e.GetPosition(this);
            var x = (int)(position.X / CellSize);
            var y = (int)(position.Y / CellSize);

            for (var i = -PenSize; i <= PenSize; i++) {
                for (var j = -PenSize; j <= PenSize; j++) {
                    var newX = x + i;
                    var newY = y + j;
                    if (newX is >= 0 and < Size && newY is >= 0 and < Size) {
                        _pixels[newX, newY].Fill = Keyboard.IsKeyDown(Key.LeftShift) || Keyboard.IsKeyDown(Key.RightShift)
                            ? _cleanColor
                            : _drawColor;
                    }
                }
            }
        }
    }

    public void Clear() {
        for (var x = 0; x < Size; x++) {
            for (var y = 0; y < Size; y++) {
                _pixels[x, y].Fill = _cleanColor;
            }
        }
    }
}
