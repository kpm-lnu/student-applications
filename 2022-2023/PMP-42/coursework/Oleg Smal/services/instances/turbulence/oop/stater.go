package oop

import (
	"github.com/Oleg-Smal-git/diploma/services/instances/turbulence"
	"github.com/Oleg-Smal-git/diploma/services/interfaces"
	oopinterfaces "github.com/Oleg-Smal-git/diploma/services/oop"
)

// Confirm that Stater satisfies oop.Stater interface.
// This will throw a compile error otherwise.
var _ oopinterfaces.Stater = (*Stater)(nil)

// Stater exports/imports the state of the simulation.
type Stater struct{}

// NewStater instantiates a turbulence stater.
func NewStater() *Stater {
	return &Stater{}
}

// Freeze exports the current state of the simulation.
func (s Stater) Freeze(oop *oopinterfaces.OOP, state interface{}) {
	castState, success := state.(*turbulence.State)
	if !success {
		panic("invalid destination state type")
	}
	for i, e := range oop.Entities {
		if i >= len(castState.Balls) {
			break
		}
		ball, success := e.(*Ball)
		if !success {
			continue
		}
		castState.Balls[i] = &turbulence.Ball{
			X:      ball.X,
			Y:      ball.Y,
			Radius: ball.Radius,
			SpeedX: ball.SpeedX,
			SpeedY: ball.SpeedY,
		}
	}
	castState.LastFrameDuration = oop.LastFrameDuration
}

// Restore sets the State and Globals of the simulation to one provided.
func (s Stater) Restore(oop *oopinterfaces.OOP, state interface{}, globals *interfaces.Globals) {
	castState, success := state.(*turbulence.State)
	if !success {
		panic("invalid destination state type")
	}
	oop.Globals = globals
	oop.Entities = make([]oopinterfaces.Entity, 0)
	collider := Collider{
		balls:    make([]*Ball, 0),
		boundary: globals.Boundary,
	}
	oop.Entities = append(oop.Entities, &collider)
	for _, e := range castState.Balls {
		b := Ball{
			X:         e.X,
			Y:         e.Y,
			Radius:    e.Radius,
			SpeedX:    e.SpeedX,
			SpeedY:    e.SpeedY,
			frameTime: globals.FrameSimulationTime,
		}
		oop.Entities = append(oop.Entities, &b)
		collider.balls = append(collider.balls, &b)
	}
}
