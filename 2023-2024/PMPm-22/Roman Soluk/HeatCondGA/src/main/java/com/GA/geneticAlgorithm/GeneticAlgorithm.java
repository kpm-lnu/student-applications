package com.GA.geneticAlgorithm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutionException;

import com.GA.Chromosome;
import com.GA.Function;
import com.GA.operation.crossover.Crossover;
import com.GA.operation.mutation.Mutation;
import com.GA.operation.selection.Selection;

public abstract class GeneticAlgorithm {

    private int generationCounter = 500;
    private int noChangeIter = 20;
    private static final Random random = new Random();
    protected Selection selection;
    protected Mutation mutation;
    protected Crossover crossover;
    protected List<Function> nonlinearSystem;
    private int populationSize;
    private int populationNumber;

    public GeneticAlgorithm(Selection selection, Mutation mutation, Crossover crossover, List<Function> nonlinearSystem) {
        this.selection = selection;
        this.mutation = mutation;
        this.crossover = crossover;
        this.nonlinearSystem = nonlinearSystem;
    }

    protected final Random gene = new Random();

    public GeneticAlgorithm() {

    }

    int getBitNum(double a, double b, double precision) {
        double length = b - a;
        long num = (long) Math.pow(10, String.valueOf(precision).length() - 1.0);
        long sizeRanges = (long) length * num;
        int i = 1;
        while (sizeRanges > Math.pow(2, i)) {
            i++;
        }
        return i;
    }


    protected Double[] getResult(double a, double b, List<Chromosome> var) {
        List<Double> varValues = new ArrayList<>();
        for (int i = 0; i < var.size(); i++) {
            varValues.add(convertToDecimal(var.get(i), a, b));
        }
        return varValues.toArray(new Double[var.size()]);
    }

    public static double convertToDecimal(Chromosome chromosome, double a, double b) {
        double xStroke = 0;

        for (int i = 0; i < chromosome.getGenes().size(); i++) {
            if (chromosome.getGenes().get(i) == 1) {
                xStroke += Math.pow(2, i);
            }
        }

        return a + xStroke * (b - a) / (Math.pow(2, chromosome.getGenes().size()));
    }

    protected boolean getProbabilityCheck(double operationProbability) {
        return operationProbability >= random.nextDouble();
    }


    protected abstract double systemGenerationMin(double a, double b, double precision, int population, double mutationProbability, double crossoverProbability);

    public abstract double executeAlgorithm(double a, double b, double precision, int[] populations, double mutationProbability, double crossoverProbability) throws ExecutionException, InterruptedException ;

    Chromosome getVariable(int startIndex, int length, List<Integer> list) {
        List<Integer> sublist = new ArrayList<>();
        for (int i = 0; i < length; i++) {
            sublist.add(list.get(i + startIndex));
        }
        return new Chromosome(sublist);
    }
}
