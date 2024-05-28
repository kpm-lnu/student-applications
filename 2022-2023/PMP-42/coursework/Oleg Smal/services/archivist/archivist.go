package archivist

import (
	"os"

	"github.com/Oleg-Smal-git/diploma/services/interfaces"
)

// Confirm that Archivist satisfies interfaces.Archivist interface.
// This will throw a compile error otherwise.
var _ interfaces.Archivist = (*Archivist)(nil)

// Archivist is an interface used to interact with disk.
type Archivist struct {
	marshal   func(v interface{}) ([]byte, error)
	unmarshal func(data []byte, v interface{}) error
}

// NewArchivist instantiate an Archivist.
func NewArchivist(marshal func(v interface{}) ([]byte, error), unmarshal func(data []byte, v interface{}) error) *Archivist {
	return &Archivist{
		marshal:   marshal,
		unmarshal: unmarshal,
	}
}

// LoadState sets the State from source file into target.
func (a *Archivist) LoadState(source string, target interface{}) error {
	data, err := os.ReadFile(source)
	if err != nil {
		return err
	}
	return a.unmarshal(data, target)
}

// SaveState saves the source State in target file.
func (a *Archivist) SaveState(target string, source interface{}) error {
	data, err := a.marshal(source)
	if err != nil {
		return err
	}
	return os.WriteFile(target, data, 0644)
}
