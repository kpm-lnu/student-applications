package com.GA.geneticAlgorithm;

import java.util.*;
import java.lang.*;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.stream.Collectors;

import com.GA.Chromosome;
import com.GA.Function;
import com.GA.HeatConductionApproximation;
import com.GA.HeatConductionFitnessFunction;
import com.GA.operation.crossover.Crossover;
import com.GA.operation.mutation.Mutation;
import com.GA.operation.selection.Selection;

public class ParallelGeneticAlgorithm extends GeneticAlgorithm {
    private int generationCounter = 500;
    private int noChangeIter = 20;
    private static final Random random = new Random();
    private Selection selection;
    private Mutation mutation;
    private Crossover crossover;

    private List<Function> nonlinearSystem;

    private HeatConductionFitnessFunction fitnessFunction;
    private HeatConductionApproximation approximation;
    protected final Random gene = new Random();

    public ParallelGeneticAlgorithm(Selection selection, Mutation mutation, Crossover crossover, List<Function> nonlinearSystem, HeatConductionFitnessFunction fitnessFunction, HeatConductionApproximation approximation) {
        super();
        //super(selection, mutation, crossover, nonlinearSystem);
        this.selection = selection;
        this.mutation = mutation;
        this.crossover = crossover;
        this.nonlinearSystem = nonlinearSystem;
        this.fitnessFunction = fitnessFunction;
        this.approximation = approximation;
    }

    @Override
    protected double systemGenerationMin(double a, double b, double precision, int population, double mutationProbability, double crossoverProbability) {
        return 0;
    }


    @Override
    public double executeAlgorithm(double a, double b, double precision, int[] populations, double mutationProbability, double crossoverProbability) throws ExecutionException, InterruptedException {
        List<Double> populationMin = new ArrayList<>();
        int bitNumber = getBitNum(a, b, precision);
        int varNum = nonlinearSystem.size();
        List<List<Chromosome>> population = selection.formPopulation(bitNumber * varNum, populations.length, populations[0]);

        ExecutorService executor = Executors.newFixedThreadPool(5);

        double generationCounter1 = 0;
        double noChangeIter1 = 0;
        List<Callable<List<Double>>> tasks = new ArrayList<>(); // List to store tasks

        for (int i = 0; i < populations.length; i++) {
            double generationMin = 0;
            List<Double> generationMinimum = new ArrayList<>();
            while (generationCounter1 < this.generationCounter && noChangeIter1 < this.noChangeIter) {
                double result = 0;

                List<Double> generationValue = new ArrayList<>();
                final List<Chromosome>[] generation = new List[]{population.get(i)};
                int populationSize = populations[i];
                for (int k = 0; k < populationSize; k++) {
                    final int chromosomeIndex = k;
                    int finalI = i;
                    List<List<Chromosome>> finalPopulation = population;
                    Callable<List<Double>> task = () -> {
                        double result1 = 0;
                        if (getProbabilityCheck(mutationProbability)) {
                            Chromosome chromosomeToMutate = generation[0].get(finalI);
                            generation[0].remove(finalI);
                            generation[0].add(mutation.mutate(chromosomeToMutate));
                        }
                        if (getProbabilityCheck(crossoverProbability)) {
                            generation[0] = crossover.crossover(generation[0], finalI, random.nextInt(populationSize));
                        }
                        for (int j = 0; j < varNum; j++) {
                            result1 = 0;
                            List<Chromosome> var = new ArrayList<>();
                            for (int v = 0; v < varNum; v++) {
                                var.add(getVariable(v * bitNumber, bitNumber, generation[0].get(finalI).getGenes()));
                            }

                            result1 += Math.abs(evaluateFitness(a, b, precision, generation));
                            generationValue.add(result1);
                        }

                        return generationValue;
                    };
                    tasks.add(task);
                }
                generationMin = Collections.min(generationValue);
                population = selection.executeModification(population, generationValue, i);

                if (generationMinimum.isEmpty()) {
                    generationMinimum.add(generationMin);
                }

                if (generationMin > Collections.min(generationMinimum)) {
                    generationMinimum.add(generationMin);
                    noChangeIter1++;
                } else {
                    generationMinimum.add(generationMin);
                    noChangeIter1 = 0;
                }
                generationCounter1++;
            }
            generationCounter1 = 0;
            noChangeIter1 = 0;
            populationMin.add(Collections.min(generationMinimum));
        }
        List<Future<List<Double>>> futures = executor.invokeAll(tasks); // Execute tasks
        List<Double> values = new ArrayList<>();
        for (Future<List<Double>> future : futures) {
            values.addAll(future.get());
        }
        return Collections.min(values);
    }




    @Override
    Chromosome getVariable(int startIndex, int length, List<Integer> list) {
        List<Integer> sublist = new ArrayList<>();
        for (int i = 0; i < length; i++) {
            sublist.add(list.get(i + startIndex));
        }
        return new Chromosome(sublist);
    }

    private Double evaluateFitness(double a, double b, double precision, List<Chromosome>[] population) {
        List<Double> fitnessValues = new ArrayList<>();
        List<Double> parameters = new Random().doubles(population[0].get(0).getSize())
                .boxed()
                .collect(Collectors.toList());

        for (Chromosome chromosome : Arrays.stream(population).collect(Collectors.toList()).get(0)) {
            double fitness = fitnessFunction.calculateFitness(parameters, Arrays.stream(population).collect(Collectors.toList()).get(0), a, b, precision);
            fitnessValues.add(fitness);
        }

        return Collections.max(fitnessValues);
    }
}