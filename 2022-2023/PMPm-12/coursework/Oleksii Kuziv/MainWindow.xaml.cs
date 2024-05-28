using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows;
using CircuitBreakerSimulator.Models;
using CircuitBreakerSimulator.Services;
using CircuitBreakerSimulator.Services.Interfaces;
using LiveCharts;
using LiveCharts.Defaults;

namespace CircuitBreakerSimulator
{
    public partial class MainWindow : Window, INotifyPropertyChanged
    {
        private CircuitBreaker _staticCircuitBreaker;
        private CircuitBreaker _adaptiveCircuitBreaker;

        private ChartValues<ObservablePoint> _staticResponseTimes;
        private ChartValues<ObservablePoint> _adaptiveResponseTimes;

        public ChartValues<ObservablePoint> StaticResponseTimes
        {
            get { return _staticResponseTimes; }
            set
            {
                _staticResponseTimes = value;
                NotifyPropertyChanged();
            }
        }

        public ChartValues<ObservablePoint> AdaptiveResponseTimes
        {
            get { return _adaptiveResponseTimes; }
            set
            {
                _adaptiveResponseTimes = value;
                NotifyPropertyChanged();
            }
        }

        public MainWindow()
        {
            InitializeComponent();
            DataContext = this;

            InitializeServices();
            InitializeCharts();
        }

        private void InitializeServices()
        {
            _staticCircuitBreaker = new StaticCircuitBreaker(5, TimeSpan.FromSeconds(10));
            _adaptiveCircuitBreaker = new AdaptiveCircuitBreaker(10);
        }

        private void InitializeCharts()
        {
            StaticResponseTimes = new ChartValues<ObservablePoint>();
            AdaptiveResponseTimes = new ChartValues<ObservablePoint>();

            chart1.Series[0].Values = StaticResponseTimes;
            chart2.Series[0].Values = AdaptiveResponseTimes;
        }

        private void Simulate()
        {
            var staticServices = new List<IFunctionService>
            {
                // new CubicFunctionService(
                //     new FunctionCoefficients { CoefficientA = 1, CoefficientB = 1, CoefficientC = 1, CoefficientD = 1 },
                //     new FunctionServiceOptions { ErrorProbability = 0.1 },
                //     _staticCircuitBreaker),
                // new ExponentialFunctionService(new FunctionCoefficients { CoefficientA = 1, CoefficientB = 1 },
                //     new FunctionServiceOptions { ErrorProbability = 0.1 },
                //     _staticCircuitBreaker),
                new LinearFunctionService(new FunctionCoefficients { CoefficientA = 1, CoefficientB = 1 },
                    new FunctionServiceOptions { ErrorProbability = 0.1 },
                    _staticCircuitBreaker),
                new QuadraticFunctionService(new FunctionCoefficients { CoefficientA = 1, CoefficientB = 1, CoefficientC = 1 },
                    new FunctionServiceOptions { ErrorProbability = 0.1 },
                    _staticCircuitBreaker)
            };

            var adaptiveServices = new List<IFunctionService>
            {
                // new CubicFunctionService(
                //     new FunctionCoefficients { CoefficientA = 1, CoefficientB = 1, CoefficientC = 1, CoefficientD = 1 },
                //     new FunctionServiceOptions { ErrorProbability = 0.1 },
                //     _adaptiveCircuitBreaker),
                // new ExponentialFunctionService(new FunctionCoefficients { CoefficientA = 1, CoefficientB = 1 },
                //     new FunctionServiceOptions { ErrorProbability = 0.1 },
                //     _adaptiveCircuitBreaker),
                new LinearFunctionService(new FunctionCoefficients { CoefficientA = 1, CoefficientB = 1 },
                    new FunctionServiceOptions { ErrorProbability = 0.1 },
                    _adaptiveCircuitBreaker),
                new QuadraticFunctionService(new FunctionCoefficients { CoefficientA = 1, CoefficientB = 1, CoefficientC = 1 },
                    new FunctionServiceOptions { ErrorProbability = 0.1 },
                    _adaptiveCircuitBreaker)
            };

            foreach (var staticService in staticServices)
            {
                var staticResponseTimes = new List<double>();

                for (int i = 0; i < 100; i++)
                {
                    var result = staticService.Calculate(i);
                    staticResponseTimes.Add(result);

                    if (staticService.GetBreaker().IsRequestAllowed())
                    {
                        staticService.GetBreaker().MarkSuccess();
                    }
                    else
                    {
                        staticService.GetBreaker().MarkFailure();
                    }
                }

                for (int i = 0; i < staticResponseTimes.Count; i++)
                {
                    StaticResponseTimes.Add(new ObservablePoint(i, staticResponseTimes[i]));
                }
            }

            foreach (var adaptiveService in adaptiveServices)
            {
                var adaptiveResponseTimes = new List<double>();

                for (int i = 0; i < 100; i++)
                {
                    var result = adaptiveService.Calculate(i);
                    adaptiveResponseTimes.Add(result);

                    if (adaptiveService.GetBreaker().IsRequestAllowed())
                    {
                        adaptiveService.GetBreaker().MarkSuccess();
                    }
                    else
                    {
                        adaptiveService.GetBreaker().MarkFailure();
                    }
                }

                for (int i = 0; i < adaptiveResponseTimes.Count; i++)
                {
                    AdaptiveResponseTimes.Add(new ObservablePoint(i, adaptiveResponseTimes[i]));
                }
            }
        }

        private void SimulateButton_Click(object sender, RoutedEventArgs e)
        {
            Simulate();
        }

        // INotifyPropertyChanged implementation
        public event PropertyChangedEventHandler PropertyChanged;

        protected virtual void NotifyPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}