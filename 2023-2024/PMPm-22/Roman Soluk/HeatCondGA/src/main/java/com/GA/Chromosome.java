package com.GA;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Chromosome {
    private List<Integer> genes;
    private static final Random gene = new Random();

    public Chromosome(List<Integer> genes){
        this.genes = genes;
    }

    public List<Integer> getGenes(){
        return genes;
    }

    public int getSize(){
        return genes.size();
    }

    public Chromosome toChromosome(List<Integer> genes){
        this.genes = genes;
        return this;
    }

    public Chromosome (int size) {
        List<Integer> bits = new ArrayList<>(size);
        byte[] array = new byte[size];

        gene.nextBytes(array);

        for (int i = 0; i < size; i++) {
            if (array[i] % 2 == 0) {
                bits.add(1);
            } else {
                bits.add(0);
            }
        }
        this.genes = bits;
    }
}
