package interfaces

import (
	"github.com/Oleg-Smal-git/gg"
)

type (
	// Runner is an interface that represents an object that performs all
	// the physics calculations, regardless of how they are implemented.
	Runner interface {
		// Next performs one atomic step of the simulation.
		Next()
		// Freeze exports the current state of the simulation.
		Freeze(interface{})
		// Restore sets the State and Globals of the simulation to one provided.
		Restore(interface{}, *Globals)
	}

	// Archivist is an interface used to interact with disk.
	Archivist interface {
		// LoadState sets the State from source file into target.
		LoadState(source string, target interface{}) error
		// SaveState saves the source State in target file.
		SaveState(target string, source interface{}) error
	}

	// Globals is a wrapper for all simulation config values, like frame duration.
	Globals struct {
		// FrameSimulationTime is the duration of an atomic simulation step.
		FrameSimulationTime float64
		// Boundary = walls :D
		Boundary struct {
			MinX, MaxX, MinY, MaxY float64
		}
	}

	// Renderer generates images out of state snapshots
	// and collects them into aggregation files.
	Renderer interface {
		// BulkRender renders all files in sourceDirectory and saves results to destinationDirectory.
		BulkRender(sourceDirectory string, destinationDirectory string, template Renderable) error
		// Collect create an aggregation file (like .gif or .mp4).
		Collect(sourceDirectory string, destination string) error
	}

	// Renderable is an interfaces that's required to be
	// satisfied by objects passed into Renderer.
	Renderable interface {
		// New instantiates an empty Renderable.
		New() Renderable
		// Render renders the object to the target context.
		Render(context *gg.Context)
	}
)
