package com.GA.operation.selection;

import java.util.ArrayList;
import java.util.List;

import com.GA.Chromosome;

public interface Selection {
    List<List<Chromosome>> formPopulation(int chromosomeSize, int populationNumber, int populationSize);
    List<List<Chromosome>> executeModification(List<List<Chromosome>> population, List<Double> fitnessValues, int index);
}
