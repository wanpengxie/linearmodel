package base

import (
	"math"
	"math/rand"
	"strings"
)

func NEQFloat(a, b float64) bool {
	eps := 1e-7
	return math.Abs(a-b) > eps
}

func NEQFloat32(a, b float32) bool {
	eps := 1e-7
	return math.Abs(float64(a-b)) > eps
}

func NEQSliceFloat32(a, b []float32) bool {
	if len(a) != len(b) {
		return true
	}
	for i := 0; i < len(a); i++ {
		if NEQFloat32(a[i], b[i]) {
			return true
		}
	}
	return false
}

func DeepCopyString(s string) string {
	var sb strings.Builder
	if _, err := sb.WriteString(s); err != nil {
	}
	return sb.String()
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
