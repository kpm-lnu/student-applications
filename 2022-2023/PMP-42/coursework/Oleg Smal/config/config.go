package config

import (
	"encoding/json"
	"github.com/Oleg-Smal-git/diploma/services/interfaces"
)

var (
	// MarshalFunctor ...
	MarshalFunctor = json.Marshal
	// UnmarshalFunctor ...
	UnmarshalFunctor = json.Unmarshal
	// StateSource is a path to initial state source file.
	StateSource = "./buff/start"
	// StateDestination is a directory to which intermediate states are saved.
	StateDestination = "./buff/state"
	// FrameDestination is a directory to which individual frames are saved.
	FrameDestination = "./buff/frames"
	// AggregationDestination is a file to which the end results are saved.
	AggregationDestination = "./buff/collection.avi"
	// MemoryProfileDestination is a file to which memory profile is written.
	MemoryProfileDestination = "./buff/memory-profile"
	// CPUProfileDestination is a file to which cpu profile is written.
	CPUProfileDestination = "./buff/cpu-profile"
	// TimeProfileDestination is a file to which frame evaluation times is written.
	TimeProfileDestination = "./buff/time-profile.txt"
	// StateCapacity describes the max amount of entities to be stored in state.
	// Used to pre-allocate memory during initialization.
	StateCapacity = 10000
	// FrameCap is the amount of frames after which the simulation stops.
	FrameCap = 100
	// FrameDuration is the amount of imaginary time that a frame lasts.
	// for simplicity's sake, all the numbers here are calibrated around
	// this variable being evaluated in seconds.
	FrameDuration = 1. / 60
	// ImageWidth describes frame size.
	ImageWidth = 1024
	// ImageHeight describes frame size.
	ImageHeight = 1024
	// FramesPerSecond is the amount of frames shown per one second of rendered animation.
	FramesPerSecond = int32(60)
	// GraphicsWorkerPool is the size of the concurrent worker bucket for renderer.
	GraphicsWorkerPool = 10
)

var (
	Globals = interfaces.Globals{
		FrameSimulationTime: FrameDuration,
		Boundary: struct{ MinX, MaxX, MinY, MaxY float64 }{
			MinX: 0,
			MaxX: float64(ImageWidth),
			MinY: 0,
			MaxY: float64(ImageHeight),
		},
	}
)
