using BoxCountingFractal.Logic;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Windows;
using ZedGraph;

namespace BoxCountingFractal
{
    public partial class GraphWindow:Window
    {
        private GraphPane graphPane;
        
        public GraphWindow()
        {
            InitializeComponent();

            graphPane = graph.GraphPane;
        }

       
        public void DrawApproximation(IList<Tuple<double, double>> points)
        {
            graphPane.Title.Text = "Графік апроксимації";
            graphPane.YAxis.Title.Text = "N(eps)";
            graphPane.XAxis.Title.Text = "eps";

            graphPane.CurveList.Clear();

            // Малюємо крапки
            PointPairList list = new PointPairList();

            foreach (Tuple<double, double> point in points)
            {
                list.Add(point.Item1, point.Item2);
            }

            LineItem scatterLine = graphPane.AddCurve("", list, Color.DarkBlue, SymbolType.Triangle);
            scatterLine.Line.IsVisible = false;
            scatterLine.Symbol.Fill.Color = Color.DarkBlue;
            scatterLine.Symbol.Fill.Type = FillType.Brush;

            // Малюємо апроксимацію
            LessSquare.GetCoefficient(points, out double k, out double b);

            PointPairList fList = new PointPairList();
            double xMin = ((int)points[0].Item1) - 1;
            double xMax = ((int)points[points.Count - 1].Item1) + 1;
            double yMin = ((int)points[0].Item2) - 1;
            double yMax = ((int)points[points.Count - 1].Item2) + 1;

            for (double x = xMin; x < xMax; x++)
            {
                fList.Add(x, LinearFunction(x, k, b));
            }

            LineItem line = graphPane.AddCurve("", fList, Color.Red, SymbolType.None);

            graphPane.XAxis.Scale.Min = xMin;
            graphPane.XAxis.Scale.Max = xMax;
            graphPane.YAxis.Scale.Min = yMin;
            graphPane.YAxis.Scale.Max = yMax;

            graph.AxisChange();
            graph.Invalidate();
        }
        public void DrawRelation(IList<Tuple<double, double>> points, string title, string yAxisTitle, string xAxisTitle)
        {
            graphPane.Title.Text = title;
            graphPane.YAxis.Title.Text = yAxisTitle;
            graphPane.XAxis.Title.Text = xAxisTitle;

            graphPane.CurveList.Clear();

            // Малюємо крапки
            PointPairList list = new PointPairList();

            foreach (Tuple<double, double> point in points)
            {
                list.Add(point.Item1, point.Item2);
            }

            LineItem scatterLine = graphPane.AddCurve("", list, Color.DarkBlue, SymbolType.Triangle);
            scatterLine.Symbol.Fill.Color = Color.DarkBlue;
            scatterLine.Symbol.Fill.Type = FillType.Brush;

            graph.AxisChange();
            graph.Invalidate();
        }
        public void DrawLayers(IList<Tuple<Tuple<double, double>, double>> points, string title, string yAxisTitle, string xAxisTitle)
        {
            graphPane.Title.Text = title;
            graphPane.YAxis.Title.Text = yAxisTitle;
            graphPane.XAxis.Title.Text = xAxisTitle;

            graphPane.CurveList.Clear();

            PointPairList list = new PointPairList();
            string[] names = new string[points.Count];

            int i = 0;
            foreach (Tuple<Tuple<double, double>, double> point in points)
            {
                list.Add(point.Item1.Item1, point.Item2);

                names[i] = string.Format("от {0} до {1} ", Math.Round(point.Item1.Item1 * 1E+5, 3), Math.Round(point.Item1.Item2 * 1E+5, 3));
                i++;
            }

            BarItem curve = graphPane.AddBar("", list, Color.Blue);

            graphPane.BarSettings.MinClusterGap = 5;
            graphPane.XAxis.Type = AxisType.Text;
            graphPane.XAxis.Scale.TextLabels = names;
            graphPane.XAxis.Scale.FontSpec.Size = 10;

            graph.AxisChange();
            graph.Invalidate();
        }
        private double LinearFunction(double x, double k, double b)
        {
            return k * x + b;
        }
    }
}
