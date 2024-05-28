package main

import (
	"github.com/Oleg-Smal-git/diploma/config"
	"github.com/Oleg-Smal-git/diploma/services/archivist"
	"github.com/Oleg-Smal-git/diploma/services/instances/turbulence"
	"github.com/Oleg-Smal-git/diploma/services/interfaces"
	"github.com/Oleg-Smal-git/diploma/services/render"
)

func initialize() interfaces.Renderer {
	return render.NewRenderer(
		archivist.NewArchivist(config.MarshalFunctor, config.UnmarshalFunctor), config.ImageWidth, config.ImageHeight,
		config.FramesPerSecond, config.GraphicsWorkerPool,
	)
}

func main() {
	// Initialize renderer.
	renderer := initialize()
	// Render individual frames.
	if err := renderer.BulkRender(config.StateDestination, config.FrameDestination, &turbulence.State{}); err != nil {
		panic(err)
	}
	// Collect frames into aggregation.
	if err := renderer.Collect(config.FrameDestination, config.AggregationDestination); err != nil {
		panic(err)
	}
}
