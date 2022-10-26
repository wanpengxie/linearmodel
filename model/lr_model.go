package model

import (
	"fmt"

	"linearmodel/base"
	"linearmodel/conf"
	"linearmodel/optim"
)

type LRModel struct {
	model       *concurrentMap
	counter     *concurrentCounter
	optim       optim.Optimizer
	conf        *conf.AllConfig
	sample      float64
	embSize     uint32
	eval        bool
	filterCount uint32
}

func (lr *LRModel) Init(conf *conf.AllConfig) error {
	lr.embSize = conf.OptimConfig.EmbSize
	lr.model = NewConcurrentMap(uint64(MODELCAP), 0)
	lr.optim = &optim.Ftrl{}
	lr.optim.Init(conf.OptimConfig)
	lr.conf = conf
	lr.filterCount = conf.FilterCount
	lr.counter = NewCounter()
	return nil
}

func (lr *LRModel) Load(path string) error {
	err := lr.model.load(path, 0)
	return err
}

func (lr *LRModel) Save(path string) error {
	metaLine := fmt.Sprintf("%d\t%d\t", 0, 0)
	n := len(lr.conf.FeatureList)
	for i, feaInfo := range lr.conf.FeatureList {
		info := fmt.Sprintf("%d:%d", feaInfo.SlotId, feaInfo.Cross)
		metaLine += info
		if i != n-1 {
			metaLine += "\t"
		}
	}
	err := lr.model.save(path, metaLine)
	return err
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

func (lr *LRModel) filterIns(ins *base.Instance) {
	for _, fea := range ins.Feas {
		key := fea.Fea
		if lr.model.exist(key) {
			continue
		}
		fea.IsFilter = !lr.counter.count(key, int(lr.filterCount))
	}
}

func (lr *LRModel) predictz(ins *base.Instance, needInit bool) float32 {
	z := lr.model.getWeight(0, 0, "", false).W
	if needInit {
		lr.filterIns(ins)
	}
	for i, n := 0, len(ins.Feas); i < n; i++ {
		fea := ins.Feas[i]
		if fea.IsFilter {
			continue
		}
		z += lr.model.getWeight(fea.Fea, fea.Slot, fea.Text, needInit).W
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
			fea := ins.Feas[j]
			if fea.IsFilter {
				continue
			}
			key := fea.Fea
			slot := fea.Slot
			lr.model.update(key, slot, ins.Label, grad, lr.optim)
		}
	}
	return nil
}

func (lr *LRModel) Eval(p bool) {
	lr.eval = p
}
