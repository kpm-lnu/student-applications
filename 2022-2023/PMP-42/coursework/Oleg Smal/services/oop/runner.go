package oop

import (
	"time"

	"github.com/Oleg-Smal-git/diploma/services/interfaces"
)

// Confirm that OOP satisfies interfaces.Runner interface.
// This will throw a compile error otherwise.
var _ interfaces.Runner = (*OOP)(nil)

// OOP is just a thing to compare ECS to, nothing more.
type OOP struct {
	Entities           []Entity
	Globals            *interfaces.Globals
	LastFrameDuration  time.Duration
	stater             Stater
	lastFrameStartTime time.Time
	lastFrameEndTime   time.Time
}

// NewOOP constructs an OOP object.
func NewOOP(stater Stater) *OOP {
	return &OOP{
		Entities: make([]Entity, 0),
		stater:   stater,
	}
}

// Next performs one atomic step of the simulation.
func (o *OOP) Next() {
	o.lastFrameStartTime = time.Now()
	// This would be a great place to introduce concurrency, but in order
	// to be able to compare this approach with others, all computations
	// are going to be performed linearly.
	for i := range o.Entities {
		o.Entities[i].Update()
	}
	o.lastFrameEndTime = time.Now()
	o.LastFrameDuration = o.lastFrameEndTime.Sub(o.lastFrameStartTime)
}

// Freeze exports the current state of the simulation.
func (o *OOP) Freeze(state interface{}) {
	o.stater.Freeze(o, state)
}

// Restore sets the state of the simulation to one provided.
func (o *OOP) Restore(state interface{}, globals *interfaces.Globals) {
	o.stater.Restore(o, state, globals)
}
