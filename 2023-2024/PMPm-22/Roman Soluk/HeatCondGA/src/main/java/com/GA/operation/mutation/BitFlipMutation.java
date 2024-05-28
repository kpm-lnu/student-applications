package com.GA.operation.mutation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.GA.Chromosome;

public class BitFlipMutation implements Mutation {
    @Override
    public Chromosome mutate(Chromosome chromosome) {
        int genesToMutateNumber = random.nextInt(1 + chromosome.getSize() / 4);
        List<Integer> mutatedChromosome = new ArrayList<>(chromosome.getSize());
        int[] genesToMutate = new int[genesToMutateNumber];

        for (int i = 0; i < genesToMutateNumber; i++) {
            genesToMutate[i] = random.nextInt(chromosome.getSize());
        }

        for (int i = 0; i < chromosome.getSize(); i++) {
            int finalI = i;
            if (Arrays.stream(genesToMutate).anyMatch(x -> x == finalI)) {
                mutatedChromosome.add(Math.abs(chromosome.getGenes().get(i) - 1));
            } else {
                mutatedChromosome.add(chromosome.getGenes().get(i));
            }
        }

        return chromosome.toChromosome(mutatedChromosome);
    }
}
