package ecs

import (
	"math"

	"github.com/Oleg-Smal-git/diploma/services/ecs"
	"github.com/Oleg-Smal-git/diploma/services/interfaces"
)

// SystemRegistrar wraps all systems to be queried during preemptive archivist allocation.
// Notice how the order in which systems are registered here is in fact the order in which
// they are going to be executed later on during the actual simulation.
var SystemRegistrar = []ecs.System{
	&SystemMover{},
	&SystemCollider{},
	&SystemBoundary{},
}

// Confirm that system structures satisfy System interface.
// This will throw a compile error otherwise.
var (
	_ ecs.System = (*SystemMover)(nil)
	_ ecs.System = (*SystemCollider)(nil)
	_ ecs.System = (*SystemBoundary)(nil)
)

// SystemMover implements the movement logic.
type SystemMover struct {
	globals  interfaces.Globals
	active   *ComponentActive
	position *ComponentPosition
	velocity *ComponentVelocity
}

// Archetype returns a minimal required bitset for the system.
func (SystemMover) Archetype() ecs.ComponentID {
	return ComponentIDActive | ComponentIDPosition | ComponentIDVelocity
}

// Run performs one atomic step of the system logic.
func (s *SystemMover) Run(index *int, entity *ecs.Entity, entities []*ecs.Entity) {
	// Assert component types and cache them in buffer.
	if s.active = entity.Components[ComponentIDActive].(*ComponentActive); !s.active.Active {
		return
	}
	s.position = entity.Components[ComponentIDPosition].(*ComponentPosition)
	s.velocity = entity.Components[ComponentIDVelocity].(*ComponentVelocity)

	// Business logic.
	s.position.X += s.velocity.X * s.globals.FrameSimulationTime
	s.position.Y += s.velocity.Y * s.globals.FrameSimulationTime
}

// New allocates all the required archivist for the System.
func (SystemMover) New() ecs.System {
	// It's okay to keep pointers as nil since those will just be redirected
	// to already existing components during Run calls.
	return &SystemMover{}
}

// Restore propagates simulation globals to child System objects.
func (s *SystemMover) Restore(globals *interfaces.Globals) {
	s.globals = *globals
}

// SystemCollider implements the collision logic.
type SystemCollider struct {
	globals   interfaces.Globals
	active    *ComponentActive
	rigidBody *ComponentRigidBody
	position  *ComponentPosition
	velocity  *ComponentVelocity

	activeIterator    *ComponentActive
	rigidBodyIterator *ComponentRigidBody
	positionIterator  *ComponentPosition
	velocityIterator  *ComponentVelocity

	velocityDifference     ComponentVelocity
	velocityExchange       ComponentVelocity
	centralDirectionVector ComponentPosition
	realignmentVector      ComponentPosition
	momentumExchangeModule float64
	centralDistance        float64
	overlap                float64
	i                      int
}

// Archetype returns a minimal required bitset for the system.
func (SystemCollider) Archetype() ecs.ComponentID {
	return ComponentIDActive | ComponentIDRigidBody | ComponentIDPosition | ComponentIDVelocity
}

func (s *SystemCollider) Run(index *int, entity *ecs.Entity, entities []*ecs.Entity) {
	// Assert component types and cache them in buffer.
	if s.active = entity.Components[ComponentIDActive].(*ComponentActive); !s.active.Active {
		return
	}
	s.rigidBody = entity.Components[ComponentIDRigidBody].(*ComponentRigidBody)
	s.position = entity.Components[ComponentIDPosition].(*ComponentPosition)
	s.velocity = entity.Components[ComponentIDVelocity].(*ComponentVelocity)

	// Business logic.
	for s.i = range entities {
		if *index <= s.i {
			continue // Skip already evaluated pairs and self.
		}
		// Assert component types and cache them in buffer.
		if s.activeIterator = (entities)[s.i].Components[ComponentIDActive].(*ComponentActive); !s.active.Active {
			continue
		}
		s.rigidBodyIterator = (entities)[s.i].Components[ComponentIDRigidBody].(*ComponentRigidBody)
		s.positionIterator = (entities)[s.i].Components[ComponentIDPosition].(*ComponentPosition)
		s.velocityIterator = (entities)[s.i].Components[ComponentIDVelocity].(*ComponentVelocity)

		// Check for collision.
		s.centralDirectionVector.X = s.positionIterator.X - s.position.X
		s.centralDirectionVector.Y = s.positionIterator.Y - s.position.Y
		s.centralDistance = math.Sqrt(math.Pow(s.centralDirectionVector.X, 2) + math.Pow(s.centralDirectionVector.Y, 2))
		if s.overlap = s.rigidBody.Size + s.rigidBodyIterator.Size - s.centralDistance; s.overlap < 0 {
			continue // Skip a pair if it's too far apart.
		}

		// Evaluate the collision.
		s.velocityDifference.X = s.velocity.X - s.velocityIterator.X
		s.velocityDifference.Y = s.velocity.Y - s.velocityIterator.Y
		s.centralDirectionVector.X /= s.centralDistance
		s.centralDirectionVector.Y /= s.centralDistance
		s.realignmentVector.X = s.centralDirectionVector.X * s.overlap / 2
		s.realignmentVector.Y = s.centralDirectionVector.Y * s.overlap / 2
		s.momentumExchangeModule = s.centralDirectionVector.X*s.velocityDifference.X + s.centralDirectionVector.Y*s.velocityDifference.Y
		s.velocityExchange.X = s.momentumExchangeModule * s.centralDirectionVector.X
		s.velocityExchange.Y = s.momentumExchangeModule * s.centralDirectionVector.Y

		// Perform the collision.
		s.velocity.X -= s.velocityExchange.X
		s.velocity.Y -= s.velocityExchange.Y
		s.velocityIterator.X += s.velocityExchange.X
		s.velocityIterator.Y += s.velocityExchange.Y
		s.position.X -= s.realignmentVector.X
		s.position.Y -= s.realignmentVector.Y
		s.positionIterator.X += s.realignmentVector.X
		s.positionIterator.Y += s.realignmentVector.Y
	}
}

// New allocates all the required archivist for the System.
func (SystemCollider) New() ecs.System {
	// It's okay to keep pointers as nil since those will just be redirected
	// to already existing components during Run calls.
	return &SystemCollider{}
}

// Restore propagates simulation globals to child System objects.
func (s *SystemCollider) Restore(globals *interfaces.Globals) {
	s.globals = *globals
}

// SystemBoundary implements the boundary logic.
type SystemBoundary struct {
	globals   interfaces.Globals
	active    *ComponentActive
	rigidBody *ComponentRigidBody
	boundary  *ComponentBoundary
	position  *ComponentPosition
	velocity  *ComponentVelocity

	delta float64
}

// Archetype returns a minimal required bitset for the system.
func (SystemBoundary) Archetype() ecs.ComponentID {
	return ComponentIDActive | ComponentIDRigidBody | ComponentIDBoundary | ComponentIDPosition | ComponentIDVelocity
}

// Run performs one atomic step of the system logic.
func (s *SystemBoundary) Run(index *int, entity *ecs.Entity, entities []*ecs.Entity) {
	// Assert component types and cache them in buffer.
	if s.active = entity.Components[ComponentIDActive].(*ComponentActive); !s.active.Active {
		return
	}
	s.rigidBody = entity.Components[ComponentIDRigidBody].(*ComponentRigidBody)
	s.boundary = entity.Components[ComponentIDBoundary].(*ComponentBoundary)
	s.position = entity.Components[ComponentIDPosition].(*ComponentPosition)
	s.velocity = entity.Components[ComponentIDVelocity].(*ComponentVelocity)

	// Business Logic.
	if s.delta = s.position.X + s.rigidBody.Size - s.boundary.MaxX; s.delta > 0 {
		s.position.X -= s.delta // Right edge.
		s.velocity.X = -math.Abs(s.velocity.X)
	}
	if s.delta = s.position.X - s.rigidBody.Size - s.boundary.MinX; s.delta < 0 {
		s.position.X -= s.delta // Left edge.
		s.velocity.X = math.Abs(s.velocity.X)
	}
	if s.delta = s.position.Y + s.rigidBody.Size - s.boundary.MaxY; s.delta > 0 {
		s.position.Y -= s.delta // Bottom edge.
		s.velocity.Y = -math.Abs(s.velocity.Y)
	}
	if s.delta = s.position.Y - s.rigidBody.Size - s.boundary.MinY; s.delta < 0 {
		s.position.Y -= s.delta // Top edge.
		s.velocity.Y = math.Abs(s.velocity.Y)
	}
}

// New allocates all the required archivist for the System.
func (SystemBoundary) New() ecs.System {
	// It's okay to keep pointers as nil since those will just be redirected
	// to already existing components during Run calls.
	return &SystemBoundary{}
}

// Restore propagates simulation globals to child System objects.
func (s *SystemBoundary) Restore(globals *interfaces.Globals) {
	s.globals = *globals
}
