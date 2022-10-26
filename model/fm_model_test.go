package model

import (
	"math/rand"
	"testing"

	"linearmodel/base"
	"linearmodel/conf"
)

func _gen_fm_instance() []*base.Instance {
	inslist := []*base.Instance{}
	ins0 := base.Instance{Feas: []*base.Feature{{Fea: 1, Slot: 101}, {Fea: 2, Slot: 103},
		{Fea: 10, Slot: 201}, {Fea: 21, Slot: 301}}, Label: 0, UserId: 1, UserIdStr: "1"}
	ins1 := base.Instance{Feas: []*base.Feature{{Fea: 1, Slot: 101}, {Fea: 3, Slot: 103},
		{Fea: 10, Slot: 201}, {Fea: 22, Slot: 301}}, Label: 1, UserId: 1, UserIdStr: "1"}

	inslist = append(inslist, &ins0)
	inslist = append(inslist, &ins1)
	return inslist
}

func _gen_fm_config() *conf.AllConfig {
	config := conf.AllConfig{}

	//conf.FeatureConfig{}
	config.FeatureList = []*conf.FeatureConfig{{SlotId: 101},
		{SlotId: 103},
		{SlotId: 201},
		{SlotId: 301},
	}
	config.OptimConfig = &conf.OptimConfig{Alpha: 1.0, Beta: 1.0, L1: 0.1, L2: 0.1,
		EmbAlpha: 0.5, EmbBeta: 0.5, EmbL1: 0.0, EmbL2: 0.2, EmbSize: 2}
	return &config
}

func TestFMModel_Predict(t *testing.T) {
	rand.Seed(0)
	insList := _gen_fm_instance()
	config := _gen_fm_config()
	fm := &FMModel{}
	fm.Init(config)
	res, _ := fm.Predict(insList)
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

func TestFMModel_Train(t *testing.T) {
	insList := _gen_fm_instance()
	config := _gen_fm_config()
	fm := &FMModel{}
	fm.Init(config)
	rand.Seed(0)
	fm.Train(insList)
	trueParameters := map[int]*base.Parameter{
		0: &base.Parameter{Slot: 0, Fea: 0, W: 0.0072583, Z: -0.1142070, N: 0.7350564},
		1: &base.Parameter{Slot: 101, Fea: 1, Text: "", W: 0.0072583, Z: -0.1142070, N: 0.7350564,
			VecW: []float32{0.1380186, 0.1311407}, VecZ: []float32{-0.1944094, -0.2095893}, VecN: []float32{0.0108758, 0.0396410}},
		2: &base.Parameter{Slot: 103, Fea: 2, Text: "", W: -0.2479608, Z: 0.4956615, N: 0.2456804,
			VecW: []float32{-0.0011852, 0.0283621}, VecZ: []float32{0.0014266, -0.0402020}, VecN: []float32{0.0000033, 0.0118219}},
		3: &base.Parameter{Slot: 103, Fea: 3, Text: "", W: 0.3331682, Z: -0.6995542, N: 0.4893760,
			VecW: []float32{-0.0269877, 0.0900312}, VecZ: []float32{0.0369334, -0.1636975}, VecN: []float32{0.0071000, 0.0955521}},
		10: &base.Parameter{Slot: 201, Fea: 10, Text: "", W: 0.0072583, Z: -0.1142070, N: 0.7350564,
			VecW: []float32{0.0210456, 0.1374886}, VecZ: []float32{-0.0337271, -0.2225331}, VecN: []float32{0.0405158, 0.0437974}},
		21: &base.Parameter{Slot: 301, Fea: 21, Text: "", W: -0.2479608, Z: 0.4956615, N: 0.2456804,
			VecW: []float32{-0.1542250, 0.2118428}, VecZ: []float32{0.2357441, -0.3895223}, VecN: []float32{0.0269901, 0.1019948}},
		22: &base.Parameter{Slot: 301, Fea: 22, Text: "", W: 0.3331682, Z: -0.6995542, N: 0.4893760,
			VecW: []float32{0.0930276, -0.0179544}, VecZ: []float32{-0.1519662, 0.0235054}, VecN: []float32{0.0469936, 0.0029794}},
	}
	if !base.EQParameter(trueParameters[0], fm.model.bias, false) {
		t.Error("bias term not the same")
	}
	for k, v := range trueParameters {
		pm := fm.model.get(uint64(k), v.Slot, false)
		if !base.EQParameter(v, pm, false) {
			t.Error("parameter not equal: ", v, " != ", pm)
		}
	}
	fm.Train(insList)
	trueParameters = map[int]*base.Parameter{
		0: &base.Parameter{Slot: 0, Fea: 0, W: 0.0023833, Z: -0.1050664, N: 1.0523415},
		1: &base.Parameter{Slot: 101, Fea: 1, Text: "", W: 0.0023833, Z: -0.1050664, N: 1.0523415,
			VecW: []float32{0.2005517, 0.0709077}, VecZ: []float32{-0.2899108, -0.1211896}, VecN: []float32{0.0150757, 0.0648001}},
		2: &base.Parameter{Slot: 103, Fea: 2, Text: "", W: -0.4780453, Z: 0.9302055, N: 0.4053446,
			VecW: []float32{-0.0027895, -0.0886123}, VecZ: []float32{0.0033622, 0.1454372}, VecN: []float32{0.0000071, 0.0486809}},
		3: &base.Parameter{Slot: 103, Fea: 3, Text: "", W: 0.5416449, Z: -1.1314875, N: 0.6469969,
			VecW: []float32{0.0514247, 0.1064395}, VecZ: []float32{-0.0765108, -0.1938382}, VecN: []float32{0.0207105, 0.0964450}},
		10: &base.Parameter{Slot: 201, Fea: 10, Text: "", W: 0.0023833, Z: -0.1050664, N: 1.0523415,
			VecW: []float32{0.0836228, 0.0772671}, VecZ: []float32{-0.1376532, -0.1329820}, VecN: []float32{0.0497556, 0.0678784}},
		21: &base.Parameter{Slot: 301, Fea: 21, Text: "", W: -0.4780453, Z: 0.9302055, N: 0.4053446,
			VecW: []float32{-0.1948737, 0.1487666}, VecZ: []float32{0.3024371, -0.2798901}, VecN: []float32{0.0309698, 0.1160778}},
		22: &base.Parameter{Slot: 301, Fea: 22, Text: "", W: 0.5416449, Z: -1.1314875, N: 0.6469969,
			VecW: []float32{0.1347290, 0.0346926}, VecZ: []float32{-0.2229768, -0.0479419}, VecN: []float32{0.0517568, 0.0082724}},
	}
	if !base.EQParameter(trueParameters[0], fm.model.bias, false) {
		t.Error("bias term not the same")
	}
	for k, v := range trueParameters {
		pm := fm.model.get(uint64(k), v.Slot, false)
		if !base.EQParameter(v, pm, false) {
			t.Error("parameter not equal: ", v, " != ", pm)
		}
	}
}
func TestFMModel_Train_Predict(t *testing.T) {
	insList := _gen_fm_instance()
	config := _gen_fm_config()
	fm := &FMModel{}
	fm.Init(config)
	rand.Seed(0)
	fm.Train(insList)
	fm.Train(insList)
	res, _ := fm.Predict(insList)
	var s0, s1 float32 = 0.2715865750551162, 0.7680124866573479
	if base.NEQFloat32(res[0].Score, s0) || base.NEQFloat32(res[1].Score, s1) {
		t.Error("predict error")
	}
}

func TestFMModel_Save_Load(t *testing.T) {
	save_path := "/tmp/fm_model"
	insList := _gen_fm_instance()
	config := _gen_fm_config()
	lr := &FMModel{}
	lr.Init(config)
	lr.Train(insList)

	err := lr.Save(save_path)
	if err != nil {
		t.Error("save file error: ", err)
	}

	new_lr := &FMModel{}
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
