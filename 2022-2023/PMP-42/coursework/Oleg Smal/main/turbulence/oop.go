//go:build BUILD_OOP

package main

import (
	"github.com/Oleg-Smal-git/diploma/config"
	"github.com/Oleg-Smal-git/diploma/services/archivist"
	oopinstances "github.com/Oleg-Smal-git/diploma/services/instances/turbulence/oop"
	"github.com/Oleg-Smal-git/diploma/services/interfaces"
	"github.com/Oleg-Smal-git/diploma/services/oop"
)

func initialize() (interfaces.Runner, interfaces.Archivist) {
	return oop.NewOOP(
		oopinstances.NewStater(),
	), archivist.NewArchivist(config.MarshalFunctor, config.UnmarshalFunctor)
}
