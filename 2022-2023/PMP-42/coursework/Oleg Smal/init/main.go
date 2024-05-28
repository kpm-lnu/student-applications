package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"

	"github.com/Oleg-Smal-git/diploma/config"
	"github.com/Oleg-Smal-git/diploma/services/archivist"
	"github.com/Oleg-Smal-git/diploma/services/instances/turbulence"
)

func main() {
	overrideN := flag.Int("overrideN", 0, "")
	flag.Parse()
	var n int
	if *overrideN != 0 {
		n = *overrideN
	} else {
		n = defaultN
	}
	area := float64((yMax - yMin) * (xMax - xMin))
	density := float64(n) / area
	radius := 1. / (4 * math.Sqrt(2) * speed * freeRun * density)
	arch := archivist.NewArchivist(config.MarshalFunctor, config.UnmarshalFunctor)
	state := turbulence.State{
		Balls:             make([]*turbulence.Ball, 0),
		LastFrameDuration: 0,
	}
	rows := int(math.Ceil(math.Sqrt(float64(n))))
	for i := 0; i < n; i++ {
		col, row := i%rows, i/rows
		state.Balls = append(state.Balls, &turbulence.Ball{
			X:      float64(xMin + (xMax-xMin)*(col+1)/(rows+1)),
			Y:      float64(yMin + (yMax-yMin)*(row+1)/(rows+1)),
			Radius: radius,
			SpeedX: 2 * speed * (rand.Float64() - 0.5),
			SpeedY: 2 * speed * (rand.Float64() - 0.5),
		})
	}
	if err := arch.SaveState(fmt.Sprintf("./buff/start"), state); err != nil {
		panic(err)
	}
}
