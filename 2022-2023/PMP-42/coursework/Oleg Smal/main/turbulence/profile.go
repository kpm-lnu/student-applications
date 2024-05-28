//go:build BUILD_PROFILE

package main

import (
	"math/rand"
	"os"
	"strconv"

	"github.com/Oleg-Smal-git/diploma/config"
	"github.com/Oleg-Smal-git/diploma/services/instances/turbulence"
	"github.com/Oleg-Smal-git/diploma/services/interfaces"

	"runtime/pprof"
)

func solve(runner interfaces.Runner, archivist interfaces.Archivist, state *turbulence.State) {
	var (
		memory, cpu, time *os.File
		err               error
		sample            int
	)
	if memory, err = os.Create(config.MemoryProfileDestination); err != nil {
		panic(err)
	}
	if cpu, err = os.Create(config.CPUProfileDestination); err != nil {
		panic(err)
	}
	if time, err = os.Create(config.TimeProfileDestination); err != nil {
		panic(err)
	}
	defer cpu.Close()
	defer time.Close()
	sample = rand.Intn(config.FrameCap) // Sample RAM at random.
	if err = pprof.StartCPUProfile(cpu); err != nil {
		panic(err)
	}
	for i := 0; i < config.FrameCap; i++ {
		runner.Next()
		runner.Freeze(state)
		time.Write([]byte(
			strconv.FormatInt(state.LastFrameDuration.Nanoseconds(), 10) + "\n",
		))
		if i == sample {
			if err = pprof.WriteHeapProfile(memory); err != nil {
				panic(err)
			}
			memory.Close()
		}
	}
	pprof.StopCPUProfile()
}
