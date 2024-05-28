package ecs

import (
	"github.com/Oleg-Smal-git/diploma/services/ecs"
	"github.com/Oleg-Smal-git/diploma/services/instances/turbulence"
	"github.com/Oleg-Smal-git/diploma/services/interfaces"
)

// Confirm that Stater satisfies ecs.Stater interface.
// This will throw a compile error otherwise.
var _ ecs.Stater = (*Stater)(nil)

// Stater exports/imports the state of the simulation.
type Stater struct {
	activeBuffer    *ComponentActive
	rigidBodyBuffer *ComponentRigidBody
	boundaryBuffer  *ComponentBoundary
	positionBuffer  *ComponentPosition
	velocityBuffer  *ComponentVelocity
	ballBuffer      *turbulence.Ball
}

// NewStater instantiates a turbulence stater.
func NewStater() *Stater {
	return &Stater{}
}

// Freeze exports the current state of the simulation.
func (s *Stater) Freeze(ecs *ecs.ECS, state interface{}) {
	castState, success := state.(*turbulence.State)
	if !success {
		panic("invalid destination state type")
	}
	for i, e := range ecs.Chunks[ArchetypeBall].Entities {
		// Cast and cache all components.
		s.activeBuffer = e.Components[ComponentIDActive].(*ComponentActive)
		if i >= len(castState.Balls) || !s.activeBuffer.Active {
			break
		}
		s.rigidBodyBuffer = e.Components[ComponentIDRigidBody].(*ComponentRigidBody)
		s.boundaryBuffer = e.Components[ComponentIDBoundary].(*ComponentBoundary)
		s.positionBuffer = e.Components[ComponentIDPosition].(*ComponentPosition)
		s.velocityBuffer = e.Components[ComponentIDVelocity].(*ComponentVelocity)
		s.ballBuffer = castState.Balls[i]
		// Set component fields.
		s.ballBuffer.Radius = s.rigidBodyBuffer.Size
		s.ballBuffer.X, s.ballBuffer.Y = s.positionBuffer.X, s.positionBuffer.Y
		s.ballBuffer.SpeedX, s.ballBuffer.SpeedY = s.velocityBuffer.X, s.velocityBuffer.Y
	}
	castState.LastFrameDuration = ecs.LastFrameDuration
}

// Restore sets the State and Globals of the simulation to one provided.
func (s *Stater) Restore(ecs *ecs.ECS, state interface{}, globals *interfaces.Globals) {
	castState, success := state.(*turbulence.State)
	if !success {
		panic("invalid source state type")
	}
	ecs.Globals = globals
	for i, e := range ecs.Chunks[ArchetypeBall].Entities {
		// Cast and cache all components.
		s.activeBuffer = e.Components[ComponentIDActive].(*ComponentActive)
		if i >= len(castState.Balls) {
			s.activeBuffer.Active = false
			break
		}
		s.rigidBodyBuffer = e.Components[ComponentIDRigidBody].(*ComponentRigidBody)
		s.boundaryBuffer = e.Components[ComponentIDBoundary].(*ComponentBoundary)
		s.positionBuffer = e.Components[ComponentIDPosition].(*ComponentPosition)
		s.velocityBuffer = e.Components[ComponentIDVelocity].(*ComponentVelocity)
		s.ballBuffer = castState.Balls[i]
		// Set component fields.
		s.activeBuffer.Active = true
		s.rigidBodyBuffer.Size = s.ballBuffer.Radius
		s.boundaryBuffer.MinX, s.boundaryBuffer.MaxX, s.boundaryBuffer.MinY, s.boundaryBuffer.MaxY =
			globals.Boundary.MinX, globals.Boundary.MaxX, globals.Boundary.MinY, globals.Boundary.MaxY
		s.positionBuffer.X, s.positionBuffer.Y = s.ballBuffer.X, s.ballBuffer.Y
		s.velocityBuffer.X, s.velocityBuffer.Y = s.ballBuffer.SpeedX, s.ballBuffer.SpeedY
	}
}
