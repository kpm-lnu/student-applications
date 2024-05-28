//go:build BUILD_ECS

package main

import (
	"runtime/debug"

	"github.com/Oleg-Smal-git/diploma/config"
	"github.com/Oleg-Smal-git/diploma/services/archivist"
	"github.com/Oleg-Smal-git/diploma/services/ecs"
	ecsinstances "github.com/Oleg-Smal-git/diploma/services/instances/turbulence/ecs"
	"github.com/Oleg-Smal-git/diploma/services/interfaces"
)

func initialize() (interfaces.Runner, interfaces.Archivist) {
	debug.SetGCPercent(0) // Disable automatic garbage collection.
	return ecs.NewRunner(
		ecsinstances.NewStater(), ecsinstances.ComponentRegistrar,
		ecsinstances.ArchetypesRegistrar, ecsinstances.SystemRegistrar,
		config.StateCapacity,
	), archivist.NewArchivist(config.MarshalFunctor, config.UnmarshalFunctor)
}
