package main

import (
	"flag"
	"time"

	"github.com/Oleg-Smal-git/diploma/config"
	"github.com/Oleg-Smal-git/diploma/services/instances/turbulence"
)

func main() {
	// Initialize the components and set initial conditions.
	runner, archivist := initialize()
	state := turbulence.State{
		Balls:             make([]*turbulence.Ball, 0, config.StateCapacity),
		LastFrameDuration: time.Duration(0),
	}
	source := flag.String("state_source", "", "")
	flag.Parse()
	if *source == "" {
		*source = config.StateSource
	}
	if err := archivist.LoadState(*source, &state); err != nil {
		panic("initialization failure: " + err.Error())
	}
	runner.Restore(&state, &config.Globals)
	// Execute the simulation.
	solve(runner, archivist, &state)
}
