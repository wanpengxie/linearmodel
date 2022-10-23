package dataloader

import (
	"testing"

	"linearmodel/base"
	"linearmodel/conf"
)

func TestDataLoaderInit(t *testing.T) {
	path := "../test/test.conf"
	dataloader := DataLoader{}
	err := dataloader.Init(path)
	if err != nil {
		t.Error("initial error: ")
	}
	if len(dataloader.featureMap) != 2 {
		t.Error("feature len not equal = ", 2)
	}
	if dataloader.isSigned != false {
		t.Error("feature sign parse error")
	}
}

func TestDataLoaderRead(t *testing.T) {
	path := "../test/test.conf"
	config := conf.ParseConf(path)
	dataloader := DataLoader{}
	dataloader.Init(path)
	train_list := config.TrainPathList

	pipe, err := dataloader.ReadFile(train_list[0])
	if err != nil {
		t.Error(err)
		return
	}
	ins := []*base.Instance{}
	for x := range pipe {
		ins = append(ins, dataloader.ParseIns(x)...)
	}
	if len(ins) != 4 {
		t.Error("parse instance incorrect")
	}
	ins0 := ins[0]
	if ins0.Label != 0 || len(ins0.Feas) != 2 || ins0.UserIdStr != "118" || ins0.ItemIdStr != "31163499" {
		t.Error("parse first instance incorrect")
	}
	if ins0.Feas[0].Slot != 101 || ins0.Feas[1].Slot != 102 {
		t.Error("parse first instance slot id error")
	}
	if ins0.Feas[0].Text != "118" || ins0.Feas[1].Text != "31163499" {
		t.Error("read string errror")
	}
}
