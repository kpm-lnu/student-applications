package com.GA;

import java.util.List;

import static com.GA.geneticAlgorithm.GeneticAlgorithm.convertToDecimal;

public class HeatConductionApproximation {

    public double[] approximateSolution(List<Double> parameters, List<Chromosome> chromosomes, double a, double b, double precision) {
            int dataSize = parameters.size();
            double[] approximatedSolution = new double[dataSize];

            for (int i = 0; i < dataSize; i++) {
                double x = parameters.get(i);
                double u = 0.0;

                for (Chromosome chromosome : chromosomes) {
                    double alpha = convertToDecimal(chromosome, 0, 1);
                    double y = chromosome.getGenes().get(0);

                    double gaussian = Math.exp(-Math.pow(x - y, 2) / (2 * precision * precision));

                    u += alpha * gaussian;
                }

                approximatedSolution[i] = u;
            }

            return approximatedSolution;
        }

        private double convertToDecimal(Chromosome chromosome, double a, double b) {
            double xStroke = 0;

            for (int i = 0; i < chromosome.getGenes().size(); i++) {
                if (chromosome.getGenes().get(i) == 1) {
                    xStroke += Math.pow(2, i);
                }
            }

            return a + xStroke * (b - a) / (Math.pow(2, chromosome.getGenes().size()));
        }
    }
