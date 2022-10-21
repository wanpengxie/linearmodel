package optim

import (
	"linearmodel/base"
	"linearmodel/conf"
)

type Optimizer interface {
	Init(config *conf.OptimConfig)
	Update(grad float32, p *base.Parameter)
	UpdateEmb(grad []float32, p *base.Parameter)
}
