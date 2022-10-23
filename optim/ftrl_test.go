package optim

import (
	"testing"

	"linearmodel/base"
	"linearmodel/conf"
)

func TestFtrl_Init(t *testing.T) {

}

func TestFtrl_Update(t *testing.T) {
	config := &conf.OptimConfig{Alpha: 0.1, Beta: 1.0, L1: 0.1, L2: 0.1,
		EmbAlpha: 0.1, EmbBeta: 1.0, EmbL1: 0.1, EmbL2: 0.1, EmbSize: 2}
	ftrl := Ftrl{}
	ftrl.Init(config)
	var g, z, n, w float32 = 0.1, 0.0, 0.0, 1.0
	var truez, truen, truew float32 = -0.90000, 0.0100000, 0.0720721
	parameter := &base.Parameter{W: w, Z: z, N: n}
	ftrl.Update(g, parameter)
	if base.NEQFloat32(parameter.Z, truez) || base.NEQFloat32(parameter.N, truen) ||
		base.NEQFloat32(parameter.W, truew) {
		t.Error("update1 ftrl error")
	}
	ftrl.Update(g, parameter)
	truez, truen, truew = -0.8298532, 0.02000000, 0.0633872
	if base.NEQFloat32(parameter.Z, truez) || base.NEQFloat32(parameter.N, truen) ||
		base.NEQFloat32(parameter.W, truew) {
		t.Error("update2 ftrl error")
	}
}

func TestFtrl_UpdateEmb(t *testing.T) {
	config := &conf.OptimConfig{Alpha: 0.1, Beta: 1.0, L1: 0.1, L2: 0.1,
		EmbAlpha: 0.1, EmbBeta: 1.0, EmbL1: 0.1, EmbL2: 0.1, EmbSize: 2}
	ftrl := Ftrl{}
	ftrl.Init(config)
	var g = []float32{0.1, 0.1}
	var z = []float32{0.0, 0.0}
	var n = []float32{0.0, 0.0}
	var w = []float32{1.0, 1.0}
	truez := []float32{-0.90000, -0.90000}
	truen := []float32{0.0100000, 0.0100000}
	truew := []float32{0.0720721, 0.0720721}
	parameter := &base.Parameter{VecW: w, VecZ: z, VecN: n}
	ftrl.UpdateEmb(g, parameter)
	if base.NEQSliceFloat32(truez, parameter.VecZ) || base.NEQSliceFloat32(truen, parameter.VecN) ||
		base.NEQSliceFloat32(truew, parameter.VecW) {
		t.Error("update1 embedding error")
	}
	truez, truen, truew = []float32{-0.8298532, -0.8298532}, []float32{0.02000000, 0.02000000}, []float32{0.0633872, 0.0633872}
	ftrl.UpdateEmb(g, parameter)
	if base.NEQSliceFloat32(truez, parameter.VecZ) || base.NEQSliceFloat32(truen, parameter.VecN) ||
		base.NEQSliceFloat32(truew, parameter.VecW) {
		t.Error("update2 embedding error")
	}
}
