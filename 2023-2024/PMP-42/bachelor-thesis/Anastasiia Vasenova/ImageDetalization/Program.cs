using AForge.Imaging.Filters;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;

namespace Diploma
{
    public static class Program
    {
        static readonly Random rand = new();
        static List<List<Gaussian>> population = new();
        static List<int> x = new();
        static List<int> y = new();
        static List<List<int>> coordinatesOfNoiseWithDublicates = new();
        static List<List<int>> coordinatesOfNoise = new();
        static List<List<Color>> pixels = new();
        static List<List<Color>> noisePixels = new();
        static readonly List<double> h = new();
        static readonly List<double> allH = new(); //minimum H_i
        static readonly List<double> meanH = new(); //mean H_i
        static readonly List<List<Gaussian>> allBestPeople= new();

        static readonly string inputImagePath = "D:\\image_without_filters6.jpg";
        static readonly string outputImagePath = "D:\\image_with_filters.jpg";
        static readonly string outputPath = "D:\\image_with_filters6.jpg";
        static readonly Bitmap inputImage = new(inputImagePath);
        static readonly Bitmap outputImage = new(outputImagePath);

        static readonly string bestHFilePath = "D:\\diploma_results\\bestH.txt";
        static readonly string meanHFilePath = "D:\\diploma_results\\meanH.txt";
        static readonly string sigmaFilePath = "D:\\diploma_results\\sigma.jpg";

        public static void WriteToFile(string filePath, List<double> data)
        {
            using (StreamWriter writer = new StreamWriter(filePath))
            {
                foreach (var value in data)
                {
                    writer.WriteLine(value);
                }
            }
        }

        public static void PartititionImageForPixels(Bitmap image1)
        {
            pixels.Clear();
            x.Clear();
            y.Clear();
            for (int i = 0; i < image1.Height; i++)
            {
                y.Add(i);
                List<Color> row = new();
                for (int j = 0; j < image1.Width; j++)
                {
                    if (i == 0)
                    {
                        x.Add(j);
                    }
                    row.Add(image1.GetPixel(j, i));
                }
                pixels.Add(row);
            }
        }

        public static void IfIsNoise(Bitmap image1, int size)
        {
            if (coordinatesOfNoise.Count > 1)
            {
                coordinatesOfNoise.Clear();
                coordinatesOfNoiseWithDublicates.Clear();
                noisePixels.Clear();
            }
            for (int i = size; i < image1.Height - size; i += (2 * size + 1))
            {
                for (int j = size; j < image1.Width - size; j += (2 * size + 1))
                {
                    //current area
                    double sumRGBACurrent = 0;
                    //surrounding area
                    double sumRGBANeighbour = 0;
                    for (int k = i - size; k < i + size + 1; k++)
                    {
                        for (int l = j - size; l < j + size + 1; l++)
                        {
                            if (k >= i - Convert.ToInt32(size / 2) && k <= i + Convert.ToInt32(size / 2) && l >= j - Convert.ToInt32(size / 2) && l <= j + Convert.ToInt32(size / 2))
                            {
                                sumRGBACurrent += pixels[k][l].R + pixels[k][l].G + pixels[k][l].B + pixels[k][l].A;
                            }
                            else
                            {
                                sumRGBANeighbour += pixels[k][l].R + pixels[k][l].G + pixels[k][l].B + pixels[k][l].A;
                            }
                        }
                    }
                    sumRGBACurrent /= Math.Pow(size, 2);
                    sumRGBANeighbour /= Math.Pow(2 * size + 1, 2) - Math.Pow(size, 2);
                    if ((Math.Abs(sumRGBACurrent / sumRGBANeighbour) < 0.8 && sumRGBACurrent < sumRGBANeighbour) || (Math.Abs(sumRGBACurrent / sumRGBANeighbour) > 1.25 && sumRGBACurrent > sumRGBANeighbour))
                    {
                        List<int> coor = new();
                        coor.Add(j);
                        coor.Add(i);
                        coordinatesOfNoiseWithDublicates.Add(coor);
                    }
                }
            }

            //REMOVING COORDINATES WITH NOISE
            coordinatesOfNoise = coordinatesOfNoiseWithDublicates.Select(sublist => string.Join(",", sublist)) 
            .Distinct()
            .Select(str => str.Split(',').Select(int.Parse).ToList()) 
            .ToList();

            x.Clear();
            y.Clear();
            for (int i = 0; i < coordinatesOfNoise.Count; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    if (j == 0)
                    {
                        x.Add(coordinatesOfNoise[i][j]);
                    }
                    else
                    {
                        y.Add(coordinatesOfNoise[i][j]);
                    }
                }
            }

            for (int i = 0; i < x.Count; i++)
            {
                List<Color> row = new();
                row.Add(image1.GetPixel(x[i], y[i]));
                noisePixels.Add(row);
            }
        }

        public static void ApplyGaussianBlur(List<Gaussian> person, int size)
        {
            using (Bitmap image = (Bitmap)Image.FromFile(inputImagePath))
            {
                for (int j = 0; j < noisePixels.Count; j++)
                {
                    Rectangle pixel = new Rectangle(person[j].X - size, person[j].Y - size, 2 * size + 1, 2 * size + 1);

                    using (Bitmap subImage = image.Clone(pixel, image.PixelFormat))
                    {
                        GaussianBlur filter = new GaussianBlur(person[j].A, size);
                        filter.ApplyInPlace(subImage);
                        using (Graphics g = Graphics.FromImage(image))
                        {
                            g.DrawImage(subImage, pixel);
                        }
                    }
                }
                Rectangle pixelImage = new Rectangle(0, 0, image.Width, image.Height);

                using (Bitmap subImage = image.Clone(pixelImage, image.PixelFormat))
                {
                    GaussianBlur filter = new GaussianBlur(0.25);
                    filter.ApplyInPlace(subImage);
                    using (Graphics g = Graphics.FromImage(image))
                    {
                        g.DrawImage(subImage, pixelImage);
                    }
                }

                //save changes on output file
                image.Save(outputPath, ImageFormat.Png);
            }

        }
        

        public static double GetRandomNumber(double minimum, double maximum)
        {
            return rand.NextDouble() * (maximum - minimum) + minimum;
        }

        public static double TargetFunction(Bitmap image1, int size) 
        {
            if (pixels.Count > 0 || x.Count > 0 || y.Count > 0)
            {
                pixels.Clear();
                x.Clear();
                y.Clear();
            }
            for (int i = 0; i < image1.Height; i++)
            {
                y.Add(i);
                List<Color> row = new();
                for (int j = 0; j < image1.Width; j++)
                {
                    if (i == 0)
                    {
                        x.Add(j);
                    }
                    row.Add(image1.GetPixel(j, i));
                }
                pixels.Add(row);
            }
            double noise = 0;
            for (int i = size; i < image1.Height - size; i += (2 * size + 1))
            {
                for (int j = size; j < image1.Width - size; j += (2 * size + 1))
                {
                    //current area
                    double sumRGBACurrent = 0;
                    //surrounding area
                    double sumRGBANeighbour = 0;
                    for (int k = i - size; k < i + size + 1; k++)
                    {
                        for (int l = j - size; l < j + size + 1; l++)
                        {
                            if (k >= i - Convert.ToInt32(size / 2) && k <= i + Convert.ToInt32(size / 2) && l >= j - Convert.ToInt32(size / 2) && l <= j + Convert.ToInt32(size / 2))
                            {
                                sumRGBACurrent += pixels[k][l].R + pixels[k][l].G + pixels[k][l].B + pixels[k][l].A;
                            }
                            else
                            {
                                sumRGBANeighbour += pixels[k][l].R + pixels[k][l].G + pixels[k][l].B + pixels[k][l].A;
                            }
                        }
                    }
                    sumRGBACurrent /= Math.Pow(size, 2);
                    sumRGBANeighbour /= Math.Pow(2 * size + 1, 2) - Math.Pow(size, 2);
                    if ((Math.Abs(sumRGBACurrent / sumRGBANeighbour) < 0.8 && sumRGBACurrent < sumRGBANeighbour) || (Math.Abs(sumRGBACurrent / sumRGBANeighbour) > 1.25 && sumRGBACurrent > sumRGBANeighbour))
                    {
                        noise += Math.Pow(sumRGBACurrent - sumRGBANeighbour, 2);
                    }
                }
            }
            return noise / (image1.Height * image1.Width / Math.Pow(2 * size + 1, 2));
        }

        public static void HeapSort(int n, int i)
        {
            int largest = i;
            int left = 2 * i + 1;
            int right = 2 * i + 2;

            if (left < n && h[left] > h[largest])
            {
                largest = left;
            }
            if (right < n && h[right] > h[largest])
            {
                largest = right;
            }
            if (largest != i)
            {
                (population[i], population[largest]) = (population[largest], population[i]);
                (h[i], h[largest]) = (h[largest], h[i]);
                HeapSort(n, largest);
            }
        }

        public static void SortByTargetFunction(int n)
        {
            for (int i = n / 2 - 1; i >= 0; i--)
            {
                HeapSort(n, i);
            }
            for (int i = n - 1; i >= 0; i--)
            {
                (population[0], population[i]) = (population[i], population[0]);
                (h[0], h[i]) = (h[i], h[0]);
                HeapSort(i, 0);
            }
        }

        public static void CreateSigmaMap(List<Gaussian> sol, int imageSizeX, int imageSizeY, string outputPath, int size)
        {
            Bitmap sigmaMap = new Bitmap(imageSizeX, imageSizeY);
            using (Graphics gfx = Graphics.FromImage(sigmaMap))
            {
                gfx.FillRectangle(Brushes.White, 0, 0, sigmaMap.Width, sigmaMap.Height);
            }
            if (sol.Count > 0)
            {
                double maxSigma = sol.Max(s => s.A);
                double minSigma = sol.Min(s => s.A);
                double range = maxSigma - minSigma;
                for (int j = 0; j < noisePixels.Count; j++)
                {
                    //normalize the sigma value to range between 8 and 255
                    int intensity = (int)((sol[j].A - minSigma) / range * (255 - 8) + 8);
                    intensity = Math.Min(255, intensity);
                    Color color = Color.FromArgb(intensity, intensity, 0, 0);
                    using (Graphics g = Graphics.FromImage(sigmaMap))
                    {
                        g.FillEllipse(new SolidBrush(color), sol[j].X - size, sol[j].Y - size, size * 2 + 1, size * 2 + 1);
                    }
                }
            }
            sigmaMap.Save(outputPath, ImageFormat.Png);
        }

        public static void GeneticAlgorithm(int m, int n, int k, int l, Bitmap image1, int size)
        {
            int s = 0; // current generation
            int gwoi = 0; // generation without imporovement

            while (s <= m && gwoi < l && noisePixels.Count > 0)
            {
                Console.WriteLine("s = " + s);
                //USING GAUSSIAN BLUR AND SORT BY TARGET FUNCTION
                double mh = 0;
                for (int i = 0; i < n; i++)
                {
                    using (Bitmap image = (Bitmap)Image.FromFile(inputImagePath))
                    {
                        for (int j = 0; j < noisePixels.Count; j++)
                        {
                            Rectangle pixel = new Rectangle(population[i][j].X - size, population[i][j].Y - size, 2 * size + 1, 2 * size + 1);
                            using (Bitmap subImage = image.Clone(pixel, image.PixelFormat))
                            {
                                GaussianBlur filter = new GaussianBlur(population[i][j].A, size);
                                filter.ApplyInPlace(subImage);
                                using (Graphics g = Graphics.FromImage(image))
                                {
                                    g.DrawImage(subImage, pixel);
                                }
                            }
                        }
                        var hi = TargetFunction(image, size);
                        mh += hi;
                        h[i] = (hi);
                    }
                }
                mh /= n;
                SortByTargetFunction(n);

                //CHECKING WHETHER RESULTS IMPROVE
                double minFitness = h[0];
                if(minFitness == 0)
                {
                    break;
                }    
                allH.Add(h[0]);
                meanH.Add(mh);
                allBestPeople.Add(population[0]);
                if (s > 0 && minFitness < allH[s - 1] && Math.Abs(minFitness - allH[s - 1]) > 0.005)
                {
                    gwoi = 0;
                }
                else
                {
                    gwoi += 1;
                }

                //CROSSING
                //90% the best with 90% the best
                List<List<Gaussian>> children = new();
                int jEnd = Convert.ToInt32(0.009 * n * k);
                for (int i = 0; i < 0.9 * (n - 0.01 * n * k); i++)
                {
                    int j = rand.Next(0, jEnd + 1); // number of person
                    List<Gaussian> child = new();

                    int chromosomeSize = rand.Next(1, noisePixels.Count + 1); //number of chromosomes, which we will take from first choosing person
                    List<int> chromosomeIndices = new();
                    for (int w = 0; w < chromosomeSize; w++)
                    {
                        chromosomeIndices.Add(rand.Next(0, noisePixels.Count)); //random index of chromosome
                    }
                    List<int> chromosomeIndicesUnique = new();
                    if (chromosomeIndices.Count > 1)
                    {
                        chromosomeIndicesUnique = (List<int>)chromosomeIndices.Distinct().ToList();
                    }
                    else 
                    {
                        chromosomeIndicesUnique = chromosomeIndices;
                    }

                    for (int w = 0; w < noisePixels.Count; w++)
                    {
                        bool UseParentJ = false;
                        for (int q = 0; q < chromosomeIndicesUnique.Count; q++)
                        {
                            if (w == chromosomeIndicesUnique[q])
                            {
                                UseParentJ = true;
                            }
                        }
                        if (UseParentJ)
                        {
                            child.Add(population[j][w]);
                        }
                        else
                        {
                            child.Add(population[i][w]);
                        }
                    }
                    children.Add(child);
                }
                //90% the best with 10% the worst
                int iStart = Convert.ToInt32(n - 0.1 * n * (1 - 0.01 * k));
                for (int i = iStart; i < n; i++)
                {
                    int j = rand.Next(0, jEnd + 1); // number of person
                    List<Gaussian> child = new();

                    int chromosomeSize = rand.Next(1, noisePixels.Count + 1); // from first choosing person
                    List<int> chromosomeIndices = new();
                    for (int w = 0; w < chromosomeSize; w++)
                    {
                        int randomNumberOfChromosome = rand.Next(0, noisePixels.Count);
                        chromosomeIndices.Add(randomNumberOfChromosome);
                    }
                    List<int> chromosomeIndicesUnique = new();
                    if (chromosomeIndices.Count > 1)
                    {
                        chromosomeIndicesUnique = (List<int>)chromosomeIndices.Distinct().ToList();
                    }
                    else
                    {
                        chromosomeIndicesUnique = chromosomeIndices;
                    }

                    for (int w = 0; w < noisePixels.Count; w++)
                    {
                        bool UseParentJ = false;
                        for (int q = 0; q < chromosomeIndicesUnique.Count; q++)
                        {
                            if (w == chromosomeIndicesUnique[q])
                            {
                                UseParentJ = true;
                            }
                        }
                        if (UseParentJ)
                        {
                            child.Add(population[j][w]);
                        }
                        else
                        {
                            child.Add(population[i][w]);
                        }
                    }
                    children.Add(child);
                }

                //MUTATION   
                for (int i = 0; i < n; i++)
                {
                    int chromosomeSizeMutation = rand.Next(1, noisePixels.Count + 1); // from first choosing person
                    List<int> chromosomeIndicesMutation = new();
                    for (int w = 0; w < chromosomeSizeMutation; w++)
                    {
                        int randomNumberOfChromosomeMutation = rand.Next(0, noisePixels.Count);
                        chromosomeIndicesMutation.Add(randomNumberOfChromosomeMutation);
                    }
                    List<int> chromosomeIndicesUniqueMutation = new();
                    if (chromosomeIndicesMutation.Count > 1)
                    {
                        chromosomeIndicesUniqueMutation = (List<int>)chromosomeIndicesMutation.Distinct().ToList();
                    }
                    else
                    {
                        chromosomeIndicesUniqueMutation = chromosomeIndicesMutation;
                    }

                    for (int w = 0; w < noisePixels.Count; w++)
                    {
                        for (int q = 0; q < chromosomeIndicesUniqueMutation.Count; q++)
                        {
                            if (w == chromosomeIndicesUniqueMutation[q])
                            {
                                int mutationCharacter = rand.Next(2);
                                int mutationPercent = rand.Next(40);
                                double val = population[i][w].A;
                                if (mutationCharacter == 0)
                                {
                                    val -= 0.01 * mutationPercent * population[i][w].A;
                                }
                                else
                                {
                                    val += 0.01 * mutationPercent * population[i][w].A;
                                }
                                if (val > 0 && val < 10)
                                {
                                    population[i][w].A = val;
                                }
                            }
                        }
                    }
                }

                int start = n - children.Count; 
                int t = 0; // index for adding children
                for (int i = start; i < start + children.Count; i++)
                {
                    population[i] = children[t];
                    ++t;
                }
                Console.WriteLine(allH[s] + "\t\t" + meanH[s]);

                children.Clear();
                s++;
            }

            SortByTargetFunction(n);
            double minValue = allH.Min();
            int minIndex = allH.IndexOf(minValue);
            Console.WriteLine("\nSolution: s = " + minIndex + ",\tallHMin = " + minValue);
            ApplyGaussianBlur(allBestPeople[minIndex], size);
            CreateSigmaMap(allBestPeople[minIndex], inputImage.Width, inputImage.Height, sigmaFilePath, size);
        }

        static void Main()
        {
            Stopwatch stopWatch = new();
            stopWatch.Start();

            int n = 100; // number of individuals(size of population)
            int k = 15; // % of elite
            int m = 30; // number of generations
            int l = 4; // number of generations without improvement

            int size = 2;
            PartititionImageForPixels(inputImage);
            IfIsNoise(inputImage, size);
            Console.WriteLine("size = " + size);


            Console.WriteLine("noisePixels.Count = " + noisePixels.Count);
            //creating a population
            for (int i = 0; i < n; i++)
            {
                List<Gaussian> person = new();
                for (int j = 0; j < noisePixels.Count; j++)
                {
                    double weight = rand.Next(1, 50);
                    Gaussian chromosome = new Gaussian(weight, x[j], y[j]);
                    person.Add(chromosome);
                }
                population.Add(person);
            }
            var hi = TargetFunction(inputImage, size);
            Console.WriteLine(hi);
            for (int i = 0; i < n; i++)
            {
                h.Add(hi); 
            }

            GeneticAlgorithm(m, n, k, l, outputImage, size);

            WriteToFile(bestHFilePath, allH);
            WriteToFile(meanHFilePath, meanH);

            stopWatch.Stop();
            TimeSpan ts = stopWatch.Elapsed;

            string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
                                                ts.Hours, ts.Minutes, ts.Seconds,
                                                ts.Milliseconds / 10);
            Console.WriteLine("\nRunTime " + elapsedTime);
        }
    }
}