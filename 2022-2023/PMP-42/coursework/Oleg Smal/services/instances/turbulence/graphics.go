package turbulence

import (
	"github.com/Oleg-Smal-git/diploma/services/interfaces"

	"github.com/Oleg-Smal-git/gg"
)

// Confirm that State satisfies interfaces.Renderable interface.
// This will throw a compile error otherwise.
var _ interfaces.Renderable = (*State)(nil)

// New instantiates an empty Renderable.
func (s *State) New() interfaces.Renderable {
	return new(State)
}

// Render renders the object to the target context.
func (s *State) Render(context *gg.Context) {
	for _, b := range s.Balls {
		context.DrawCircle(b.X, b.Y, b.Radius)
		context.SetRGB(1, 1, 1)
		context.Fill()
	}
}
