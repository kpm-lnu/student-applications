//go:build BUILD_EXPORT

package main

import (
	"errors"
	"fmt"
	"os"

	"github.com/Oleg-Smal-git/diploma/config"
	"github.com/Oleg-Smal-git/diploma/services/instances/turbulence"
	"github.com/Oleg-Smal-git/diploma/services/interfaces"
)

func solve(runner interfaces.Runner, archivist interfaces.Archivist, state *turbulence.State) {
	if _, err := os.Stat(config.StateDestination); errors.Is(err, os.ErrNotExist) {
		err = os.Mkdir(config.StateDestination, os.ModePerm)
		if err != nil {
			panic("initialization failure: " + err.Error())
		}
	}
	// Copy initial state as first result.
	if err := archivist.SaveState(fmt.Sprintf("%v/%v", config.StateDestination, 0), state); err != nil {
		panic("archivist failure: " + err.Error())
	}
	for i := 0; i < config.FrameCap; i++ {
		runner.Next()
		runner.Freeze(state)
		if err := archivist.SaveState(fmt.Sprintf("%v/%v", config.StateDestination, i+1), *state); err != nil {
			panic("archivist failure: " + err.Error())
		}
	}
}
