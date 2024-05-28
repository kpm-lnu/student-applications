package com.GA.operation.selection;

import java.util.ArrayList;
import java.util.List;

import com.GA.Chromosome;

public class TournamentSelection implements Selection {
    @Override
    public List<List<Chromosome>> formPopulation(int chromosomeSize, int populationNumber, int populationSize) {
        List<List<Chromosome>> population = new ArrayList<>();
        for (int j = 0; j < populationNumber; j++) {
            List<Chromosome> generation = new ArrayList<>();
            for (int i = 0; i < populationSize; i++) {
                generation.add(new Chromosome(chromosomeSize));
            }
            population.add(generation);
        }
        return population;
    }

    @Override
    public List<List<Chromosome>> executeModification(List<List<Chromosome>> population, List<Double> fitnessValues, int index) {
        return population;
    }
}
