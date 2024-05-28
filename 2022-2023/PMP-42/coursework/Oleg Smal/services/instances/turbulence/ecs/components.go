package ecs

import (
	"github.com/Oleg-Smal-git/diploma/services/ecs"
)

// ComponentRegistrar wraps all components to be queried during preemptive archivist allocation.
var ComponentRegistrar = []ecs.Component{
	&ComponentActive{},
	&ComponentRigidBody{},
	&ComponentBoundary{},
	&ComponentPosition{},
	&ComponentVelocity{},
}

// Components are stored in a bitset, which means that
// each one should be of kind 2^n where n ∈ N_0+.
const (
	// ComponentIDActive is used to simulate nullability. This allows the garbage collector to
	// allocate archivist before the simulation starts to run, thus
	// significantly reducing the time that would have been spent on that otherwise.
	ComponentIDActive ecs.ComponentID = 1 << iota
	// ComponentIDRigidBody describes static object properties.
	ComponentIDRigidBody
	// ComponentIDBoundary describes existence boundaries.
	ComponentIDBoundary
	// ComponentIDPosition describes object position.
	ComponentIDPosition
	// ComponentIDVelocity describes object velocity.
	ComponentIDVelocity
)

// Confirm that all component structures satisfy Component interface.
// This will throw a compile error otherwise.
var (
	_ ecs.Component = (*ComponentActive)(nil)
	_ ecs.Component = (*ComponentRigidBody)(nil)
	_ ecs.Component = (*ComponentBoundary)(nil)
	_ ecs.Component = (*ComponentPosition)(nil)
	_ ecs.Component = (*ComponentVelocity)(nil)
)

// ComponentActive fakes nullability.
type ComponentActive struct {
	Active bool
}

// ID identifies the component.
func (ComponentActive) ID() ecs.ComponentID {
	return ComponentIDActive
}

// New allocates all the required archivist for the Component.
func (ComponentActive) New() ecs.Component {
	return &ComponentActive{}
}

// ComponentRigidBody describes static object properties.
type ComponentRigidBody struct {
	Size float64
}

// ID identifies the component.
func (ComponentRigidBody) ID() ecs.ComponentID {
	return ComponentIDRigidBody
}

// New allocates all the required archivist for the Component.
func (ComponentRigidBody) New() ecs.Component {
	return &ComponentRigidBody{}
}

// ComponentBoundary describes existence boundaries.
type ComponentBoundary struct {
	MinX, MaxX, MinY, MaxY float64
}

// ID identifies the component.
func (ComponentBoundary) ID() ecs.ComponentID {
	return ComponentIDBoundary
}

// New allocates all the required archivist for the Component.
func (ComponentBoundary) New() ecs.Component {
	return &ComponentBoundary{}
}

// ComponentPosition describes object position.
type ComponentPosition struct {
	X, Y float64
}

// ID identifies the component.
func (ComponentPosition) ID() ecs.ComponentID {
	return ComponentIDPosition
}

// New allocates all the required archivist for the Component.
func (ComponentPosition) New() ecs.Component {
	return &ComponentPosition{}
}

// ComponentVelocity describes object velocity.
type ComponentVelocity struct {
	X, Y float64
}

// ID identifies the component.
func (ComponentVelocity) ID() ecs.ComponentID {
	return ComponentIDVelocity
}

// New allocates all the required archivist for the Component.
func (ComponentVelocity) New() ecs.Component {
	return &ComponentVelocity{}
}
