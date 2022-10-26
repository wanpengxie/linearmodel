package base

import (
	"github.com/dgryski/go-farm"
)

const KMAXSIGN = 1e16

type Feature struct {
	Fea  uint64
	Slot uint16
	Text string
}

func (f *Feature) Encode() {
	fea := farm.Hash64WithSeed([]byte(f.Text), uint64(f.Slot))
	if fea >= KMAXSIGN {
		fea = fea >> 11
	}
	f.Fea = uint64(f.Slot)*KMAXSIGN + fea
}

func (f *Feature) ExtractSlot() uint16 {
	return uint16(f.Fea / KMAXSIGN)
}

type Instance struct {
	Label     int
	Len       int
	UserId    uint64
	ItemId    uint64
	UserIdStr string
	ItemIdStr string
	Feas      []*Feature
}

type Result struct {
	Label  int
	Score  float32
	UserId uint64
}

type Parameter struct {
	Slot  uint16
	Fea   uint64
	Text  string
	Show  int
	Click int

	// weight term
	W float32
	Z float32
	N float32

	// vector term
	VecW []float32
	VecN []float32
	VecZ []float32
}

type Weight struct {
	W    float32
	VecW []float32
}

func NewParameter(size uint32, norm float32) *Parameter {
	return &Parameter{W: 0.0, Z: 0.0, N: 0.0,
		VecW: RandVec32(size, norm),
		VecZ: make([]float32, size),
		VecN: make([]float32, size),
	}
}
