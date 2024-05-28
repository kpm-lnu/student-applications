package oop

import (
	"github.com/Oleg-Smal-git/diploma/services/interfaces"
)

type (
	// Entity represent a simulation entity.
	Entity interface {
		// Update is invoked every frame.
		Update()
	}
	// Stater is used to inject archive functionality into Runner.
	Stater interface {
		// Freeze exports the current state of the simulation.
		Freeze(*OOP, interface{})
		// Restore sets the State and Globals of the simulation to one provided.
		Restore(*OOP, interface{}, *interfaces.Globals)
	}
)
