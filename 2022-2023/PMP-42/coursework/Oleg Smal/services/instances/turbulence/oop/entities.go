package oop

import (
	"math"

	"github.com/Oleg-Smal-git/diploma/services/interfaces"
	"github.com/Oleg-Smal-git/diploma/services/oop"
)

// Confirm that Ball and Collider satisfy oop.Entity interface.
// This will throw a compile error otherwise.
var _ oop.Entity = (*Ball)(nil)
var _ oop.Entity = (*Collider)(nil)

// Ball goes bounce :)
type Ball struct {
	X, Y           float64
	Radius         float64
	SpeedX, SpeedY float64
	frameTime      float64
}

// Update is invoked every frame.
func (b *Ball) Update() {
	b.X += b.SpeedX * b.frameTime
	b.Y += b.SpeedY * b.frameTime
}

// Collider handles collision logic.
type Collider struct {
	// balls is an array of ball references.
	balls []*Ball
	// boundary is a copy of simulation boundary.
	boundary struct {
		MinX, MaxX, MinY, MaxY float64
	}
	globals *interfaces.Globals
}

// Update is invoked every frame.
func (c *Collider) Update() {
	for i, ei := range c.balls {
		// Boundaries collision.
		if delta := ei.X + ei.Radius - c.boundary.MaxX; delta > 0 {
			ei.X -= delta // Right edge.
			ei.SpeedX = -math.Abs(ei.SpeedX)
		}
		if delta := ei.X - ei.Radius - c.boundary.MinX; delta < 0 {
			ei.X -= delta // Left edge.
			ei.SpeedX = math.Abs(ei.SpeedX)
		}
		if delta := ei.Y + ei.Radius - c.boundary.MaxY; delta > 0 {
			ei.Y -= delta // Bottom edge.
			ei.SpeedY = -math.Abs(ei.SpeedY)
		}
		if delta := ei.Y - ei.Radius - c.boundary.MinY; delta < 0 {
			ei.Y -= delta // Top edge.
			ei.SpeedY = math.Abs(ei.SpeedY)
		}
		for j, ej := range c.balls {
			if j <= i {
				continue // Skip already evaluated pairs and self.
			}
			// Check for collision.
			deltaX, deltaY := ej.X-ei.X, ej.Y-ei.Y
			delta := math.Sqrt(math.Pow(deltaX, 2) + math.Pow(deltaY, 2))
			overlap := ei.Radius + ej.Radius - delta
			if overlap < 0 {
				continue // Skip a pair if it's too far apart.
			}

			// Evaluate the collision.
			deltaSpeedX, deltaSpeedY := ei.SpeedX-ej.SpeedX, ei.SpeedY-ej.SpeedY
			deltaX, deltaY = deltaX/delta, deltaY/delta
			correctionX, correctionY := deltaX*overlap/2, deltaY*overlap/2
			momentumExchange := deltaX*deltaSpeedX + deltaY*deltaSpeedY
			speedExchangeX, speedExchangeY := momentumExchange*deltaX, momentumExchange*deltaY

			// Perform the collision.
			ei.SpeedX -= speedExchangeX
			ei.SpeedY -= speedExchangeY
			ej.SpeedX += speedExchangeX
			ej.SpeedY += speedExchangeY
			ei.X -= correctionX
			ei.Y -= correctionY
			ej.X += correctionX
			ej.Y += correctionY
		}
	}
}
