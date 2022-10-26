package model

import (
	"math/rand"
	"testing"

	"linearmodel/base"
	"linearmodel/conf"
)

func _gen_ffm_instance() []*base.Instance {
	inslist := []*base.Instance{}
	ins0 := base.Instance{Feas: []*base.Feature{{Fea: 1, Slot: 101}, {Fea: 2, Slot: 103},
		{Fea: 10, Slot: 201}, {Fea: 21, Slot: 301}}, Label: 0, UserId: 1, UserIdStr: "1"}
	ins1 := base.Instance{Feas: []*base.Feature{{Fea: 1, Slot: 101}, {Fea: 3, Slot: 103},
		{Fea: 10, Slot: 201}, {Fea: 22, Slot: 301}}, Label: 1, UserId: 1, UserIdStr: "1"}

	inslist = append(inslist, &ins0)
	inslist = append(inslist, &ins1)
	return inslist
}

func _gen_ffm_config() *conf.AllConfig {
	config := conf.AllConfig{}

	//conf.FeatureConfig{}
	config.FeatureList = []*conf.FeatureConfig{
		{SlotId: 101, Cross: 1},
		{SlotId: 103, Cross: 2},
		{SlotId: 201, Cross: 3},
		{SlotId: 301, Cross: 3},
	}
	config.OptimConfig = &conf.OptimConfig{Alpha: 1.0, Beta: 1.0, L1: 0.1, L2: 0.1,
		EmbAlpha: 0.5, EmbBeta: 0.5, EmbL1: 0.0, EmbL2: 0.2, EmbSize: 2}
	return &config
}

func TestFFMModel_Init(t *testing.T) {
	//insList := _gen_ffm_instance()
	config := _gen_ffm_config()
	ffm := new(FFMModel)
	err := ffm.Init(config)
	if err != nil {
		t.Error("initial error: ", err)
	}
	if ffm.num_of_field != 3 {
		t.Error("field number error")
	}
	if int(ffm.full_size) != int(ffm.num_of_field)*int(config.OptimConfig.EmbSize) {
		t.Error("full size error")
	}
}

func TestFFMModel_Train(t *testing.T) {
	rand.Seed(0)
	insList := _gen_ffm_instance()
	config := _gen_ffm_config()
	ffm := new(FFMModel)
	err := ffm.Init(config)
	if err != nil {
		t.Error("initial error: ", err)
	}
	ffm.Train(insList)
	trueParameters := map[int]*base.Parameter{
		0: &base.Parameter{Slot: 0, Fea: 0, W: 0.0000000, Z: -0.0946457, N: 0.7094239},
		1: &base.Parameter{Slot: 101, Fea: 1, Text: "", W: 0.0000000, Z: -0.0946457, N: 0.7094239,
			VecW: []float32{-0.0000000, -0.0000000, -0.0041510, 0.0053728, -0.0708619, -0.0150433},
			VecZ: []float32{0.0000000, 0.0000000, 0.0053271, -0.0067200, 0.0942655, 0.0184541},
			VecN: []float32{0.0000000, 0.0000000, 0.0017357, 0.0006438, 0.0042426, 0.0001787}},
		2: &base.Parameter{Slot: 103, Fea: 2, Text: "", W: -0.2493148, Z: 0.4985395, N: 0.2485417,
			VecW: []float32{-0.0116540, 0.0305691, -0.0000000, -0.0000000, 0.0022107, 0.0240754},
			VecZ: []float32{0.0142869, -0.0389468, 0.0000000, 0.0000000, -0.0026655, -0.0301714},
			VecN: []float32{0.0001679, 0.0013712, 0.0000000, 0.0000000, 0.0000082, 0.0007076}},
		3: &base.Parameter{Slot: 103, Fea: 3, Text: "", W: 0.3254194, Z: -0.6788830, N: 0.4608822,
			VecW: []float32{0.0107175, -0.0063148, -0.0000000, -0.0000000, -0.0219625, 0.0054689},
			VecZ: []float32{-0.0131737, 0.0076814, 0.0000000, 0.0000000, 0.0276485, -0.0066368},
			VecN: []float32{0.0002128, 0.0000674, 0.0000000, 0.0000000, 0.0008672, 0.0000459}},
		10: &base.Parameter{Slot: 201, Fea: 10, Text: "", W: 0.0000000, Z: -0.0946457, N: 0.7094239,
			VecW: []float32{-0.0127167, 0.0101726, 0.0312821, -0.0341451, -0.0175717, 0.0026763},
			VecZ: []float32{0.0160424, -0.0125704, -0.0392395, 0.0433090, 0.0218256, -0.0033519},
			VecN: []float32{0.0009461, 0.0003190, 0.0007392, 0.0011690, 0.0004428, 0.0006866}},
		21: &base.Parameter{Slot: 301, Fea: 21, Text: "", W: -0.2493148, Z: 0.4985395, N: 0.2485417,
			VecW: []float32{0.0098649, 0.0155773, 0.0135415, -0.0270545, -0.0182581, 0.0286080},
			VecZ: []float32{-0.0120550, -0.0192377, -0.0167256, 0.0342754, 0.0227431, -0.0366236},
			VecN: []float32{0.0001210, 0.0003060, 0.0003086, 0.0011191, 0.0005208, 0.0016074}},
		22: &base.Parameter{Slot: 301, Fea: 22, Text: "", W: 0.3254194, Z: -0.6788830, N: 0.4608822,
			VecW: []float32{-0.0258588, -0.0031720, 0.0147715, -0.0053721, -0.0003862, 0.0077477},
			VecZ: []float32{0.0325161, 0.0038293, -0.0183388, 0.0065225, 0.0004638, -0.0094504},
			VecN: []float32{0.0008251, 0.0000130, 0.0004306, 0.0000499, 0.0000002, 0.0000977}},
	}
	if !base.EQParameter(trueParameters[0], ffm.model.bias, false) {
		t.Error("bias term not the same")
	}
	for k, v := range trueParameters {
		pm := ffm.model.get(uint64(k), v.Slot, false)
		if !base.EQParameter(v, pm, false) {
			t.Error("parameter not equal: ", v, " != ", pm)
		}
	}

	ffm.Train(insList)
	trueParameters = map[int]*base.Parameter{
		0: &base.Parameter{Slot: 0, Fea: 0, W: 0.0073989, Z: -0.1156009, N: 1.0171366},
		1: &base.Parameter{Slot: 101, Fea: 1, Text: "", W: 0.0073989, Z: -0.1156009, N: 1.0171366,
			VecW: []float32{-0.0000000, -0.0000000, 0.0026682, -0.0058631, -0.0754537, -0.0191139},
			VecZ: []float32{0.0000000, 0.0000000, -0.0034265, 0.0073640, 0.1004348, 0.0235937},
			VecN: []float32{0.0000000, 0.0000000, 0.0017740, 0.0007836, 0.0042954, 0.0002954}},
		2: &base.Parameter{Slot: 103, Fea: 2, Text: "", W: -0.4681827, Z: 0.9078129, N: 0.3911533,
			VecW: []float32{-0.0103755, 0.0289767, -0.0000000, -0.0000000, -0.0115029, 0.0422665},
			VecZ: []float32{0.0127215, -0.0369213, 0.0000000, 0.0000000, 0.0141985, -0.0536986},
			VecN: []float32{0.0001704, 0.0013753, 0.0000000, 0.0000000, 0.0002948, 0.0012418}},
		3: &base.Parameter{Slot: 103, Fea: 3, Text: "", W: 0.5402715, Z: -1.1217566, N: 0.6259833,
			VecW: []float32{0.0104786, -0.0075906, -0.0000000, -0.0000000, -0.0074352, -0.0098212},
			VecZ: []float32{-0.0128801, 0.0092355, 0.0000000, 0.0000000, 0.0094389, 0.0121811},
			VecN: []float32{0.0002129, 0.0000698, 0.0000000, 0.0000000, 0.0012074, 0.0004056}},
		10: &base.Parameter{Slot: 201, Fea: 10, Text: "", W: 0.0073989, Z: -0.1156009, N: 1.0171366,
			VecW: []float32{-0.0137402, 0.0072609, 0.0235187, -0.0395513, -0.0121565, -0.0034161},
			VecZ: []float32{0.0178547, -0.0090171, -0.0295689, 0.0502656, 0.0151262, 0.0042942},
			VecN: []float32{0.0024725, 0.0004381, 0.0008195, 0.0012566, 0.0004904, 0.0008132}},
		21: &base.Parameter{Slot: 301, Fea: 21, Text: "", W: -0.4681827, Z: 0.9078129, N: 0.3911533,
			VecW: []float32{0.0311392, 0.0201706, 0.0128657, -0.0342171, -0.0129390, 0.0278186},
			VecZ: []float32{-0.0391690, -0.0249467, -0.0158913, 0.0434329, 0.0161418, -0.0356136},
			VecN: []float32{0.0008372, 0.0003382, 0.0003093, 0.0012017, 0.0005648, 0.0016084}},
		22: &base.Parameter{Slot: 301, Fea: 22, Text: "", W: 0.5402715, Z: -1.1217566, N: 0.6259833,
			VecW: []float32{-0.0480811, -0.0108089, 0.0076047, -0.0035429, -0.0044267, 0.0057764},
			VecZ: []float32{0.0615860, 0.0131866, -0.0094692, 0.0043040, 0.0053555, -0.0070492},
			VecN: []float32{0.0016353, 0.0000998, 0.0005102, 0.0000549, 0.0000241, 0.0001035}},
	}
	if !base.EQParameter(trueParameters[0], ffm.model.bias, false) {
		t.Error("bias term not the same")
	}
	for k, v := range trueParameters {
		pm := ffm.model.get(uint64(k), v.Slot, false)
		if !base.EQParameter(v, pm, false) {
			t.Error("parameter not equal: ", v, " != ", pm)
		}
	}
}

func TestFFMModel_Train_Predict(t *testing.T) {

}

func TestFFMModel_Save_Load(t *testing.T) {

}
