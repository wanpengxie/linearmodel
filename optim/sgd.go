package optim

import (
	"linearmodel/base"
	"linearmodel/conf"
)

type SGD struct {
	alpha float32
	beta  float32
	l1    float32
	l2    float32
}

func (sgd *SGD) Init(conf *conf.OptimConfig) {
	sgd.alpha = float32(conf.Alpha)
	sgd.beta = float32(conf.Beta)
	sgd.l1 = float32(conf.L1)
	sgd.l2 = float32(conf.L2)
}

func (sgd *SGD) Update(grad float32, parameter *base.Parameter) {
	parameter.W = (1.0-2.0*sgd.l2*sgd.alpha)*parameter.W - grad*sgd.alpha
}

func (sgd *SGD) UpdateEmb(gradVec []float32, parameter *base.Parameter) {
	for i := 0; i < len(parameter.VecW); i++ {
		grad := gradVec[i]
		parameter.VecW[i] = (1.0-2.0*sgd.l2*sgd.alpha)*parameter.VecW[i] - grad*sgd.alpha
	}
}
