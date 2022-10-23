package optim

import (
	"github.com/golang/glog"

	"linearmodel/base"
	"linearmodel/conf"
)

type Ftrl struct {
	alpha    float32
	beta     float32
	l1       float32
	l2       float32
	embAlpha float32
	embBeta  float32
	embL1    float32
	embL2    float32
	embSize  uint32
}

func (ftrl *Ftrl) Init(conf *conf.OptimConfig) {
	ftrl.alpha = conf.Alpha
	ftrl.beta = conf.Beta
	ftrl.l1 = conf.L1
	ftrl.l2 = conf.L2
	ftrl.embAlpha = conf.EmbAlpha
	ftrl.embBeta = conf.EmbBeta
	ftrl.embL1 = conf.EmbL1
	ftrl.embL2 = conf.EmbL2
	ftrl.embSize = conf.EmbSize
}

func (ftrl *Ftrl) Update(grad float32, parameter *base.Parameter) {
	parameter.Z, parameter.N, parameter.W = ftrl.update(grad, parameter.Z, parameter.N, parameter.W)
}

func (ftrl *Ftrl) UpdateEmb(gradVec []float32, parameter *base.Parameter) {
	if len(gradVec) != len(parameter.VecW) || len(gradVec) != len(parameter.VecN) || len(gradVec) != len(parameter.VecZ) {
		glog.Fatalf("update embedding error, size not equal: grad=%d, emb_w=%d, emb_n=%d, emb_z=%d",
			len(gradVec), len(parameter.VecW), len(parameter.VecN), len(parameter.VecZ))
	}
	for i := 0; i < len(parameter.VecW); i++ {
		z, n, w, g := parameter.VecZ[i], parameter.VecN[i], parameter.VecW[i], gradVec[i]
		parameter.VecZ[i], parameter.VecN[i], parameter.VecW[i] = ftrl.update(g, z, n, w)
	}
}

func (ftrl *Ftrl) update(grad, z, n, w float32) (float32, float32, float32) {
	sigma := (sqrt32(n+grad*grad) - sqrt32(n)) / ftrl.alpha
	z += grad - sigma*w
	n += grad * grad
	sgn := float32(1.0)
	if z < 0 {
		sgn = -1.0
	}
	if sgn*z < ftrl.l1 {
		w = 0
	} else {
		w = -(z - sgn*ftrl.l1) / ((ftrl.beta+sqrt32(n))/ftrl.alpha + ftrl.l2)
	}
	return z, n, w
}
