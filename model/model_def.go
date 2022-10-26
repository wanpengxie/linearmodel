package model

import (
	"linearmodel/base"
	"linearmodel/conf"
)

var (
	MODELCAP = 100000
)

const (
	concurrentCount = 61
)

type IModel interface {
	Init(config *conf.AllConfig) error
	Train([]*base.Instance) error
	Predict([]*base.Instance) ([]base.Result, error)
	Eval(p bool)
	Load(path string) error
	Save(path string) error
}
