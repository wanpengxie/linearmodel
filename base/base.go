package base

import (
	"math"
	"math/rand"
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
	Label     int
	Len       int
	UserId    uint64
	ItemId    uint64
	UserIdStr string
	ItemIdStr string
	Feas      []Feature
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

func DeepCopyString(s string) string {
	var sb strings.Builder
	if _, err := sb.WriteString(s); err != nil {
	}
	return sb.String()
}

func NewParameter(size uint32) *Parameter {
	return &Parameter{W: 0.0, Z: 0.0, N: 0.0,
		VecW: RandVec32(size),
		VecZ: make([]float32, size),
		VecN: make([]float32, size),
	}
}

func RandVec32(n uint32) []float32 {
	if n == 0 {
		return []float32{}
	}
	vec := make([]float32, n, n)
	for i := uint32(0); i < n; i++ {
		vec[i] = float32((rand.Float64() - 0.5)) / float32(n)
	}
	return vec
}

func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func Sigmoid32(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(-float64(x))))
}

func VecNorm(v []float64) float64 {
	s := 0.0
	for i, n := 0, len(v); i < n; i++ {
		s += v[i] * v[i]
	}
	return s
}

func VecNorm32(v []float32) float32 {
	s := float32(0.0)
	for i, n := 0, len(v); i < n; i++ {
		s += v[i] * v[i]
	}
	return s
}

func InPlaceVecTimeAdd(v1, v2 []float32, a1, a2 float32) []float32 {
	if v1 == nil || len(v1) == 0 {
		v1 = make([]float32, len(v2))
	}

	n1, n2 := len(v1), len(v2)
	for i := 0; i < n1 && i < n2; i++ {
		v1[i] = v1[i]*a1 + v2[i]*a2
	}
	return v1
}
