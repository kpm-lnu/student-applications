package ecs

import (
	"time"

	"github.com/Oleg-Smal-git/diploma/services/interfaces"
)

// Confirm that ECS satisfies interfaces.Runner interface.
// This will throw a compile error otherwise.
var _ interfaces.Runner = (*ECS)(nil)

// ECS stands for Entity Component System and is an architectural
// pattern we will be using for this implementation of interfaces.interfaces.
type ECS struct {
	Chunks             map[ComponentID]Chunk
	Globals            *interfaces.Globals
	LastFrameDuration  time.Duration
	stater             Stater
	lastFrameStartTime time.Time
	lastFrameEndTime   time.Time
}

// NewRunner constructs an ECS object.
func NewRunner(stater Stater, componentRegistrar []Component, archetypesRegistrar []ComponentID, systemRegistrar []System, capacity int) *ECS {
	ecs := ECS{
		Chunks: make(map[ComponentID]Chunk, len(archetypesRegistrar)),
		stater: stater,
	}
	for _, a := range archetypesRegistrar {
		chunk := Chunk{
			Archetype: a,
			Entities:  make([]*Entity, capacity),
			Systems:   make([]System, 0, len(systemRegistrar)),
		}
		for i := range chunk.Entities {
			chunk.Entities[i] = new(Entity)
			chunk.Entities[i].Components = make(map[ComponentID]Component, len(componentRegistrar))
			for _, c := range componentRegistrar {
				if a&c.ID() == c.ID() {
					// This is done in order to deep copy the interface value.
					chunk.Entities[i].Components[c.ID()] = c.New()
				}
			}
		}
		for _, s := range systemRegistrar {
			if a&s.Archetype() == s.Archetype() {
				// This is done in order to deep copy the interface value.
				chunk.Systems = append(chunk.Systems, s.New())
			}
		}
		ecs.Chunks[a] = chunk
	}
	return &ecs
}

// Next performs one atomic step of the simulation.
func (r *ECS) Next() {
	r.lastFrameStartTime = time.Now()
	// This would be a great place to introduce concurrency, but in order
	// to be able to compare this approach with others, all computations
	// are going to be performed linearly.
	for _, c := range r.Chunks {
		for _, s := range c.Systems {
			for i := range c.Entities {
				s.Run(&i, c.Entities[i], c.Entities)
			}
		}
	}
	r.lastFrameEndTime = time.Now()
	r.LastFrameDuration = r.lastFrameEndTime.Sub(r.lastFrameStartTime)
}

// Freeze exports the current state of the simulation.
func (r *ECS) Freeze(state interface{}) {
	r.stater.Freeze(r, state)
}

// Restore sets the state of the simulation to one provided.
func (r *ECS) Restore(state interface{}, globals *interfaces.Globals) {
	for i := range r.Chunks {
		for j := range r.Chunks[i].Systems {
			r.Chunks[i].Systems[j].Restore(globals)
		}
	}
	r.stater.Restore(r, state, globals)
}
