package model

import (
	"testing"

	"linearmodel/base"
	"linearmodel/conf"
)

func _gen_lr_instance() []*base.Instance {
	inslist := []*base.Instance{}
	ins0 := base.Instance{Feas: []*base.Feature{{Fea: 1, Slot: 101}, {Fea: 2, Slot: 103},
		{Fea: 10, Slot: 201}, {Fea: 21, Slot: 301}}, Label: 0, UserId: 1, UserIdStr: "1"}
	ins1 := base.Instance{Feas: []*base.Feature{{Fea: 1, Slot: 101}, {Fea: 3, Slot: 103},
		{Fea: 10, Slot: 201}, {Fea: 22, Slot: 301}}, Label: 1, UserId: 1, UserIdStr: "1"}

	inslist = append(inslist, &ins0)
	inslist = append(inslist, &ins1)
	return inslist
}

func _gen_lr_config() *conf.AllConfig {
	config := conf.AllConfig{}

	//conf.FeatureConfig{}
	config.FeatureList = []*conf.FeatureConfig{{SlotId: 101},
		{SlotId: 103},
		{SlotId: 201},
		{SlotId: 301},
	}
	config.OptimConfig = &conf.OptimConfig{Alpha: 1.0, Beta: 1.0, L1: 0.1, L2: 0.1,
		EmbAlpha: 0.5, EmbBeta: 0.5, EmbL1: 0.0, EmbL2: 0.2, EmbSize: 0}
	return &config
}

func TestLRModel_Predict(t *testing.T) {
	insList := _gen_lr_instance()
	config := _gen_lr_config()
	lr := &LRModel{}
	lr.Init(config)
	res, _ := lr.Predict(insList)
	if len(res) != 2 {
		t.Error("result len: ", len(res), " error")
	}
	if base.NEQFloat32(res[0].Score, 0.5) || base.NEQFloat32(res[1].Score, 0.5) {
		t.Error("score error")
	}
	if res[0].UserId != 1 || res[1].UserId != 1 {
		t.Error("score error")
	}
}

func TestLRModel_Train(t *testing.T) {
	insList := _gen_lr_instance()
	config := _gen_lr_config()
	lr := &LRModel{}
	lr.Init(config)
	lr.Train(insList)
	trueParameters := map[int]*base.Parameter{
		0:  &base.Parameter{Slot: 0, Fea: 0, W: 0.0, Z: -0.0933346, N: 0.711284},
		1:  &base.Parameter{Slot: 101, Fea: 1, Text: "", W: 0, Z: -0.0933346, N: 0.711284},
		2:  &base.Parameter{Slot: 103, Fea: 2, Text: "", W: -0.25, Z: 0.5, N: 0.25},
		3:  &base.Parameter{Slot: 103, Fea: 3, Text: "", W: 0.3255315, Z: -0.679179, N: 0.461284},
		10: &base.Parameter{Slot: 201, Fea: 10, Text: "", W: 0.0, Z: -0.0933346, N: 0.711284},
		21: &base.Parameter{Slot: 301, Fea: 21, Text: "", W: -0.25, Z: 0.5, N: 0.25},
		22: &base.Parameter{Slot: 301, Fea: 22, Text: "", W: 0.3255315, Z: -0.679179, N: 0.461284},
	}
	if !base.EQParameter(trueParameters[0], lr.model.bias, false) {
		t.Error("bias term not the same")
	}
	for k, v := range trueParameters {
		pm := lr.model.get(uint64(k), v.Slot, false)
		if !base.EQParameter(v, pm, false) {
			t.Error("parameter not equal: ", v, " != ", pm)
		}
	}
	lr.Train(insList)
	trueParameters = map[int]*base.Parameter{
		0:  &base.Parameter{Slot: 0, Fea: 0, W: 0.0069274222, Z: -0.11461359, N: 1.01914525},
		1:  &base.Parameter{Slot: 101, Fea: 1, Text: "", W: 0.0069274222, Z: -0.11461359, N: 1.01914525},
		2:  &base.Parameter{Slot: 103, Fea: 2, Text: "", W: -0.4686705, Z: 0.90917259, N: 0.39253696},
		3:  &base.Parameter{Slot: 103, Fea: 3, Text: "", W: 0.540484068, Z: -1.12237206, N: 0.6266083},
		10: &base.Parameter{Slot: 201, Fea: 10, Text: "", W: 0.0069274222, Z: -0.11461359, N: 1.01914525},
		21: &base.Parameter{Slot: 301, Fea: 21, Text: "", W: -0.4686705, Z: 0.90917259, N: 0.39253696},
		22: &base.Parameter{Slot: 301, Fea: 22, Text: "", W: 0.540484068, Z: -1.12237206, N: 0.6266083},
	}
	if !base.EQParameter(trueParameters[0], lr.model.bias, false) {
		t.Error("bias term not the same")
	}
	for k, v := range trueParameters {
		pm := lr.model.get(uint64(k), v.Slot, false)
		if !base.EQParameter(v, pm, false) {
			t.Error("parameter not equal: ", v, " != ", pm)
		}
	}
}

func TestLRModel_TRAIN_PREDICT(t *testing.T) {
	insList := _gen_lr_instance()
	config := _gen_lr_config()
	lr := &LRModel{}
	lr.Init(config)
	lr.Train(insList)
	lr.Train(insList)
	res, _ := lr.Predict(insList)
	var predict0, predict1 float32 = 0.285659596575785, 0.7505879345496209
	if base.NEQFloat32(res[0].Score, predict0) || base.NEQFloat32(res[1].Score, predict1) {
		t.Error("predict after train error")
	}
}

func TestLRModel_Save_LOAD(t *testing.T) {
	save_path := "/tmp/lr_model"
	insList := _gen_lr_instance()
	config := _gen_lr_config()
	lr := &LRModel{}
	lr.Init(config)
	lr.Train(insList)

	err := lr.Save(save_path)
	if err != nil {
		t.Error("save file error: ", err)
	}

	new_lr := &LRModel{}
	new_lr.Init(config)
	err = new_lr.Load(save_path)
	if err != nil {
		t.Error("load file error: ", err)
	}

	if base.NEQFloat32(lr.model.bias.W, new_lr.model.bias.W) {
		t.Error("save or load bias error")
	}

	for _, x := range lr.model.modelData {
		for k, v := range x.data {
			new_pm := new_lr.model.get(k, v.Slot, false)
			if new_pm == nil {
				t.Error("load error, missing key: ", k)
				continue
			}
			if base.NEQFloat32(v.W, new_pm.W) || base.NEQSliceFloat32(v.VecW, new_pm.VecW) {
				t.Error("save or load error: ", k, v, new_pm)
			}
		}
	}
}
