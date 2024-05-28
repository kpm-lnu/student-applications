package com.GA;

import java.util.List;
import java.util.function.Function;

import static com.GA.geneticAlgorithm.GeneticAlgorithm.convertToDecimal;

public class HeatConductionFitnessFunction {

    private List<Function> nonlinearSystem;

    public HeatConductionFitnessFunction(List<Function> nonlinearSystem) {
        this.nonlinearSystem = nonlinearSystem;
    }

    public double calculateFitness(List<Double> parameters, List<Chromosome> chromosomes, double a, double b, double precision) {
        double[] approximatedSolution = getApproximatedSolution(parameters, chromosomes, a, b, precision);
        double fitness = calculateFitnessValue(approximatedSolution);

        return fitness;
    }

    private double[] getApproximatedSolution(List<Double> parameters, List<Chromosome> chromosomes, double a, double b, double precision) {
        int dataSize = parameters.size();
        int populationSize = chromosomes.size();

        double[] approximatedValues = new double[dataSize];

        for (int dataIndex = 0; dataIndex < dataSize; dataIndex++) {
            double x = parameters.get(dataIndex);

            double sum = 0.0;
            for (int i = 0; i < populationSize; i++) {
                double alpha = convertToDecimal(chromosomes.get(i), 0, 1);
                double sigma = convertToDecimal(chromosomes.get(i), 1, 2);
                double gaussian = Math.exp(-Math.pow((x - alpha), 2) / (2 * Math.pow(sigma, 2)));

                sum += gaussian;
            }
            approximatedValues[dataIndex] = sum;
        }

        return approximatedValues;

    }

    private double calculateFitnessValue(double[] approximatedSolution) {
        int dataSize = approximatedSolution.length;
        double fitnessValue = 0.0;

        for (int i = 0; i < dataSize; i++) {
            double realSolution = (double) nonlinearSystem.get(i).apply(0);
            fitnessValue += Math.pow(approximatedSolution[i] - realSolution, 2);
        }

        return fitnessValue;
    }
}
