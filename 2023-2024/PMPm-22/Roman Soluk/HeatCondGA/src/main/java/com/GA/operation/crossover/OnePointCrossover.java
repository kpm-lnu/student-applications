package com.GA.operation.crossover;

import java.util.ArrayList;
import java.util.List;

import com.GA.Chromosome;

public class OnePointCrossover implements Crossover {
    @Override
    public List<Chromosome> crossover(List<Chromosome> chromosomes, int...chromosomeIndexes) {
        int position = random.nextInt(chromosomes.get(0).getSize());
        List<Integer> crossChromo1 = new ArrayList<>();
        List<Integer> crossChromo2 = new ArrayList<>();
        for (int i = 0; i < chromosomes.get(0).getSize(); i++) {
            if (i < position) {
                crossChromo1.add(chromosomes.get(chromosomeIndexes[0]).getGenes().get(i));
                crossChromo2.add(chromosomes.get(chromosomeIndexes[1]).getGenes().get(i));
            } else {
                crossChromo1.add(chromosomes.get(chromosomeIndexes[1]).getGenes().get(i));
                crossChromo2.add(chromosomes.get(chromosomeIndexes[0]).getGenes().get(i));
            }
        }
        chromosomes.set(chromosomeIndexes[0], new Chromosome(crossChromo1));
        chromosomes.set(chromosomeIndexes[1], new Chromosome(crossChromo2));
        return chromosomes;
    }
}
