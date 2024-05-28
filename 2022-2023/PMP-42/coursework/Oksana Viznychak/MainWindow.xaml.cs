using BoxCountingFractal.Logic;
using Microsoft.Win32;
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Windows;
using System.Windows.Media.Imaging;

namespace BoxCountingFractal
{
   
    public partial class MainWindow : Window
    {
        private string imageFilepath;
        private FractalDimension fractalDimension;

        public MainWindow()
        {
            InitializeComponent();

            fractalDimension = new FractalDimension();
            imageFilepath = string.Empty;
            thresholdBlackColorNumeric.Value = fractalDimension.BlackBoundary;

            SetEnablePanels(false);
        }
        private void SetEnablePanels(bool isEnable)
        {
            CapacitiveDimensionPanel.IsEnabled = MinkowskiDimensionsPanel.IsEnabled =
                RenyiSpectrumPanel.IsEnabled = MultifactorSpectrumPanel.IsEnabled = isEnable;
        }
        private void LoadImageButtonClick(object sender, RoutedEventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog
            {
                InitialDirectory = "C:\\",
                Filter = "Image Files(*.BMP;*.JPG;*.GIF;*.PNG)|*.BMP;*.JPG;*.GIF;*.PNG",
                RestoreDirectory = true
            };

            if (openFileDialog.ShowDialog() == true)
            {
                // Отримати шлях до вказаного файлу
                imageFilepath = openFileDialog.FileName;

                BitmapImage bitmap = new BitmapImage();
                bitmap.BeginInit();
                bitmap.UriSource = new Uri(imageFilepath);
                bitmap.EndInit();

                mainImage.Source = bitmap;

                thresholdBlackColorNumeric.IsEnabled = true;
                thresholdBlackColorNumeric.Maximum = (int?)(Math.Min(mainImage.Source.Width, mainImage.Source.Height) / 16);
                SetEnablePanels(true);
            }
        }
        private void ThresholdCalculateClick(object sender, RoutedEventArgs e)
        {
            fractalDimension.BlackBoundary = (int)thresholdBlackColorNumeric.Value;
            thresholdResult.Text = fractalDimension.CalculateCapacitiveDimension(imageFilepath).ToString();

            GraphWindow graphWindow = new GraphWindow();
            graphWindow.DrawApproximation(fractalDimension.CDPoints);
            graphWindow.Show();
        }
        private void CellSizeCalculateClick(object sender, RoutedEventArgs e)
        {
            int epsilon = (int)cellSizeInput.Value;

            cellSizeResult.Text = fractalDimension.CalculateMinkowskiDimensionForGrayscaleImages(imageFilepath, epsilon).ToString();
        }

        private void ApplyThresholdClick(object sender, RoutedEventArgs e)
        {
            SetBlackBoundaryToImageBox();
        }
        private void DependenceDeltaOnCellSizeClick(object sender, RoutedEventArgs e)
        {
            int epsilon = (int)cellSizeInput.Value;

            fractalDimension.CalculateMinkowskiDimensionForGrayscaleImages(imageFilepath);

            GraphWindow graphWindow = new GraphWindow();
            graphWindow.DrawRelation(fractalDimension.MDPoints, "Графік залежності A delta від розміру осередку розбиття", "A delta", "eps");
            graphWindow.Show();
        }
        private void SetBlackBoundaryToImageBox()
        {
            int blackBoundary = (int)thresholdBlackColorNumeric.Value;
            Bitmap image = new Bitmap(Image.FromFile(imageFilepath));
            Bitmap newImage = new Bitmap(image.Width, image.Height);

            for (int x = 0; x < newImage.Width; x++)
            {
                for (int y = 0; y < newImage.Height; y++)
                {
                    Color pixel = image.GetPixel(x, y);

                    if (pixel.R <= blackBoundary && pixel.G <= blackBoundary && pixel.B <= blackBoundary)
                    {
                        newImage.SetPixel(x, y, Color.Black);
                    }
                    else
                    {
                        newImage.SetPixel(x, y, Color.White);
                    }
                }
            }

            mainImage.Source = ToBitmapImage(newImage);
        }
        public BitmapImage ToBitmapImage(Bitmap bitmap)
        {
            using (var memory = new MemoryStream())
            {
                bitmap.Save(memory, ImageFormat.Png);
                memory.Position = 0;

                var bitmapImage = new BitmapImage();
                bitmapImage.BeginInit();
                bitmapImage.StreamSource = memory;
                bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
                bitmapImage.EndInit();
                bitmapImage.Freeze();

                return bitmapImage;
            }
        }
        private void CalculateRenyiClick(object sender, RoutedEventArgs e)
        {
            fractalDimension.QMin = (int)qMinInput.Value;
            fractalDimension.QMax = (int)qMaxInput.Value;
            fractalDimension.CalculateRenyiSpectre(imageFilepath);

            GraphWindow graphWindow = new GraphWindow();
            graphWindow.DrawRelation(fractalDimension.SRPoints, "Графік залежності D(q) від значення q", "D(q)", "q");
            graphWindow.Show();
        }
        private void CalculateBoundsClick(object sender, RoutedEventArgs e)
        {
            fractalDimension.PrecalculateAlpaMinAlpaMax(imageFilepath);

            fractalMinInput.Text = String.Format("{0} * E-5", Math.Round(fractalDimension.AlphaMin * 1E+5, 3));
            fractalMaxInput.Text = String.Format("{0} * E-5", Math.Round(fractalDimension.AlphaMax * 1E+5, 3));

            splitStepInput.IsEnabled = calculateRenyiMF.IsEnabled = true;
        }
        private void CalculateMultifractalClick(object sender, RoutedEventArgs e)
        {
            fractalDimension.MFEpsilon = Convert.ToDouble(splitStepInput.Text);

            fractalDimension.CalculateMultufractalWithLocalDensityFunction();

            //малюємо графік
            GraphWindow graphWindow = new GraphWindow();
            graphWindow.DrawLayers(fractalDimension.MFPoints, "Гістограма спектрів", "Ємнісна розмірність", "Діапазони alpha (*E-5)");
            graphWindow.Show();

            //отримуємо сформовані картинки
            SubsetsWindow subsetsWindow = new SubsetsWindow(fractalDimension.LocalDensityImagesDirectory, fractalDimension.AlphaGroups);
            subsetsWindow.Show();
        }
    }
}
