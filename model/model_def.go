package model

import (
	"linearmodel/base"
	"linearmodel/conf"
)

var (
	MODELCAP = 33554432
)

const (
	concurrentCount = 99991
)

type IModel interface {
	Init(config *conf.AllConfig) error
	Train([]*base.Instance) error
	Predict([]*base.Instance) ([]base.Result, error)
	Eval(p bool)
	Load(path string) error
	Save(path string) error
}
