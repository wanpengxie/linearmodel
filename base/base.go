package base

import (
	"strings"

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
	Label  int
	Len    int
	UserId uint64
	ItemId uint64
	Feas   []Feature
}

type Result struct {
	Label  int
	Score  float64
	UserId uint64
}

type Parameter struct {
	Slot  uint16
	Fea   uint64
	Text  string
	Show  int
	Click int

	// weight term
	W float64
	Z float64
	N float64

	// vector term
	VecW []float64
	VecN []float64
	VecZ []float64
}

type ParameterW struct {
	W    float64
	VecW []float64
}

func DeepCopyString(s string) string {
	var sb strings.Builder
	if _, err := sb.WriteString(s); err != nil {
	}
	return sb.String()
}
