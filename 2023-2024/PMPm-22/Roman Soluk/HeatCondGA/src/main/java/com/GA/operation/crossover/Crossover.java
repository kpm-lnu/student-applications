package com.GA.operation.crossover;

import java.util.List;
import java.util.Random;

import com.GA.Chromosome;

public interface Crossover {
    Random random = new Random();
    public List<Chromosome> crossover(List<Chromosome> chromosomes, int...chromosomeIndexes);
}
