using System;
using System.Collections.Generic;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;

namespace BoxCountingFractal
{
   public partial class SubsetsWindow:Window
    {
        private int imageSize = 300;

        public SubsetsWindow()
        {
            InitializeComponent();
        }
        public SubsetsWindow(string imagesPath, List<Tuple<double, double>> alphaGroups)
        {
            InitializeComponent();

            var files = Directory.EnumerateFiles(imagesPath);
            int i = 0;
            foreach (string file in files)
            {
                Image image = new Image
                {
                    Source = new BitmapImage(new Uri(file)),

                    Width = imageSize,
                    Height = imageSize,
                };

                TextBlock titleBlock = new TextBlock
                {
                    Text = String.Format("від {0} до {1} ", Math.Round(alphaGroups[i].Item1 * 1E+5, 3), Math.Round(alphaGroups[i].Item2 * 1E+5, 3)),
                    TextAlignment = TextAlignment.Center,
                    VerticalAlignment = VerticalAlignment.Bottom
                };

                Grid grid = new Grid
                {
                    Width = imageSize,
                    Height = imageSize + 30
                };
                grid.Children.Add(image);
                grid.Children.Add(titleBlock);

                layoutPanel.Children.Add(grid);

                i++;
            }
        }
        private void Window_Closed(object sender, EventArgs e)
        {
            //отримати всі картинки та звільнити їх
            foreach (Grid grid in layoutPanel.Children)
            {
                foreach (UIElement control in grid.Children)
                {
                    // якщо це картинка, то видаляєм, щоб не накопичувати в пам"яті. (також може бути текст)
                    if (control.GetType().Name == "Image")
                    {
                        ((Image)control).Source = null;
                    }
                }
            }
        }
    }
}
