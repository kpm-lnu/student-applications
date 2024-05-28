package turbulence

import (
	"time"
)

type (
	// State represents an exhaustive description of physics state of the simulation.
	State struct {
		// Balls is a collection of all Ball objects that take part in the simulation.
		Balls []*Ball
		// LastFrameDuration is the amount of time it took to compute last frame.
		LastFrameDuration time.Duration
	}

	// Ball goes bounce :)
	Ball struct {
		// These properties exhaustively describes the state of the ball.
		X, Y           float64
		Radius         float64
		SpeedX, SpeedY float64
	}
)
