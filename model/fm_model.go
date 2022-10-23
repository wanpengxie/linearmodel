package model

import (
	"linearmodel/base"
	"linearmodel/conf"
	"linearmodel/optim"
)

type FMModel struct {
	model   *concurrentMap
	optim   optim.Optimizer
	conf    *conf.AllConfig
	sample  float64
	embSize uint32
	eval    bool

	group_sparse bool
}

func (fm *FMModel) Init(conf *conf.AllConfig) error {
	fm.embSize = conf.OptimConfig.EmbSize
	fm.model = NewConcurrentMap(uint64(MODELCAP), fm.embSize)
	fm.optim = &optim.Ftrl{}
	fm.optim.Init(conf.OptimConfig)
	fm.conf = conf
	return nil
}

func (fm *FMModel) Load(path string) error {
	return nil
}

func (fm *FMModel) Save(path string) error {
	return nil
}

func (fm *FMModel) Predict(inslist []*base.Instance) ([]base.Result, error) {
	n := len(inslist)
	res := make([]base.Result, n, n)
	for i := 0; i < n; i++ {
		ins := inslist[i]
		x, _, _ := fm.predict_(ins, false)
		res[i] = base.Result{UserId: ins.UserId, Label: ins.Label, Score: x}
	}
	return res, nil
}

func (fm *FMModel) Train(inslist []*base.Instance) error {
	n := len(inslist)
	for i := 0; i < n; i++ {
		ins := inslist[i]
		_, grad, gradVec := fm.predict_(ins, true)
		m := len(ins.Feas)
		fm.model.update(0, 0, ins.Label, grad, fm.optim)
		for j := 0; j < m; j++ {
			fea := ins.Feas[j]
			key := fea.Fea
			slot := fea.Slot
			//fm.model.update(key, slot, ins.Label, grad, fm.optim)
			//fm.model.updateEmb(key, slot, ins.Label, gradVec[j], fm.optim)
			fm.model.updateWeightAndEmb(key, slot, ins.Label, grad, gradVec[j], fm.optim)
		}
	}
	return nil
}

func (fm *FMModel) predict_(ins *base.Instance, needInit bool) (float32, float32, [][]float32) {
	z := fm.model.getWeight(0, 0, "", false).W
	sumVec := make([]float32, fm.embSize)
	gradVec := make([][]float32, 0, len(ins.Feas))
	for i, n := 0, len(ins.Feas); i < n; i++ {
		fea := ins.Feas[i].Fea
		slot := ins.Feas[i].Slot
		text := ins.Feas[i].Text
		factor := fm.model.getWeight(fea, slot, text, needInit)
		z += (factor.W - base.VecNorm32(factor.VecW)/2.0)
		sumVec = base.InPlaceVecTimeAdd(sumVec, factor.VecW, 1.0, 1.0)
		gradVec = append(gradVec, factor.VecW)
	}
	z += base.VecNorm32(sumVec) / 2.0
	p := base.Sigmoid32(z)
	grad := p
	if ins.Label > 0 {
		grad -= 1.0
	}
	if needInit {
		for _, gradv := range gradVec {
			for j := range gradv {
				gradv[j] = grad * (sumVec[j] - gradv[j])
			}
		}
	}
	return p, grad, gradVec
}

func (fm *FMModel) Eval(p bool) {
	fm.eval = p
}
