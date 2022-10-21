package model

import (
	"linearmodel/base"
	"linearmodel/conf"
	"linearmodel/optim"
)

type LRModel struct {
	model   *concurrentMap
	optim   optim.Optimizer
	conf    conf.AllConfig
	sample  float64
	embSize uint32
	eval    bool
}

func (lr *LRModel) Init(conf conf.AllConfig) error {
	lr.embSize = conf.OptimConfig.EmbSize
	lr.model = NewConcurrentMap(uint64(MODELCAP), lr.embSize)
	lr.optim = &optim.Ftrl{}
	lr.optim.Init(conf.OptimConfig)
	lr.conf = conf
	return nil
}

func (lr *LRModel) Load(path string) error {
	//err := lr.model.load(path)
	//return err
	return nil
}

func (lr *LRModel) Save(path string) error {
	//err := lr.model.save(path)
	return nil
}

func (lr *LRModel) Load_INC(path string) error {
	return nil
	//err := lr.model.(*concurrentMap).load_inc(path)
	//return err
}

func (lr *LRModel) Save_INC(path string) error {
	return nil
	//err := lr.model.(*concurrentMap).save_inc(path)
	//return err
}

func (lr *LRModel) Predict(inslist []*base.Instance) ([]base.Result, error) {
	n := len(inslist)
	res := make([]base.Result, n, n)
	for i := 0; i < n; i++ {
		ins := inslist[i]
		res[i] = base.Result{UserId: ins.UserId, Label: ins.Label, Score: lr.predictz(ins, false)}
	}
	return res, nil
}

func (lr *LRModel) predictz(ins *base.Instance, needInit bool) float32 {
	z := lr.model.getWeight(0, 0, false).W
	for i, n := 0, len(ins.Feas); i < n; i++ {
		fea := ins.Feas[i]
		z += lr.model.getWeight(fea.Fea, fea.Slot, needInit).W
	}
	return base.Sigmoid32(z)
}

func (lr *LRModel) gradient(ins *base.Instance) float32 {
	// g = p - y
	g := lr.predictz(ins, true)
	if ins.Label > 0 {
		g -= 1.0
	}
	return g
}

func (lr *LRModel) Train(inslist []*base.Instance) error {
	n := len(inslist)
	for i := 0; i < n; i++ {
		ins := inslist[i]
		grad := lr.gradient(ins)
		m := len(ins.Feas)
		lr.model.update(0, 0, ins.Label, grad, lr.optim)
		for j := 0; j < m; j++ {
			key := ins.Feas[j].Fea
			slot := ins.Feas[j].Slot
			lr.model.update(key, slot, ins.Label, grad, lr.optim)
		}
	}
	return nil
}

func (lr *LRModel) Eval(p bool) {
	lr.eval = p
}
