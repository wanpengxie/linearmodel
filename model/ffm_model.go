package model

import (
	"fmt"

	"github.com/golang/glog"

	"linearmodel/base"
	"linearmodel/conf"
	"linearmodel/optim"
)

type FFMModel struct {
	model *concurrentMap
	optim *optim.Ftrl
	conf  *conf.AllConfig

	// new field info
	field_vec     []int16
	slot_to_field map[uint64]int16
	num_of_field  int
	emb_size      uint32
	full_size     uint32
	slot_map      map[uint16]bool
	using_string  bool
	is_recall     bool

	// query and index part
	slot_to_side          map[uint64]int
	field_to_side         map[uint64]int
	decompose_index_table map[int][2]int
}

//
func (ffm *FFMModel) Init(conf *conf.AllConfig) error {
	ffm.optim = new(optim.Ftrl)
	ffm.optim.Init(conf.OptimConfig)
	ffm.conf = conf

	ffm.field_vec = []int16{}
	ffm.slot_to_field = make(map[uint64]int16)
	ffm.slot_to_side = make(map[uint64]int)
	ffm.field_to_side = make(map[uint64]int)
	ffm.decompose_index_table = make(map[int][2]int)
	ffm.emb_size = conf.OptimConfig.EmbSize

	slot_dict := make(map[int16]bool)
	slot_max := 0
	ffm.slot_map = make(map[uint16]bool)
	slot_list := []uint64{}
	for _, x := range conf.FeatureList {
		slot := x.SlotId
		slot_list = append(slot_list, slot)
		if x.Cross > 0 {
			slot_dict[int16(x.Cross)] = true
			if int(x.Cross) > slot_max {
				slot_max = int(x.Cross)
			}
		}
		field := x.Cross
		side := int(x.VecType)
		ffm.slot_to_field[slot] = int16(field)
		// check consist
		if _, ok := ffm.slot_to_side[slot]; ok {
			glog.Fatal("error config: dedup slot")
		}
		if s, ok := ffm.field_to_side[uint64(field)]; ok && s != side {
			glog.Fatal("error config: same field with different side")
		}
		ffm.slot_to_side[slot] = side
		ffm.field_to_side[uint64(field)] = side

		ffm.slot_map[uint16(slot)] = true
	}

	ffm.num_of_field = len(slot_dict)
	if slot_max > ffm.num_of_field {
		glog.Fatalf("max slot %d more than field num %d", slot_max, ffm.num_of_field)
	}
	for i := 0; i < slot_max; i++ {
		if slot_dict[int16(i)] {
			ffm.field_vec = append(ffm.field_vec, int16(i))
		}
	}
	ffm.full_size = conf.OptimConfig.EmbSize * uint32(ffm.num_of_field)
	//ffm.model.norm = float64(conf.FtrlConfig.FmSize) * float64(ffm.num_of_field)

	glog.Info("number of field: ", ffm.num_of_field)
	glog.Info("emb size: ", ffm.full_size)

	// make dot type vector map
	index := 0
	for i := 0; i < ffm.num_of_field; i++ {
		for j := i + 1; j < ffm.num_of_field; j++ {
			ffm.decompose_index_table[index] = [2]int{i, j}
			index += 1
		}
	}

	ffm.model = NewConcurrentMap(uint64(MODELCAP), ffm.full_size)
	return nil
}

func (ffm *FFMModel) Save(path string) error {
	err := ffm.model.load(path, int(ffm.full_size))
	return err
}

func (ffm *FFMModel) Load(path string) error {
	metaLine := fmt.Sprintf("%d\t%d\t", ffm.emb_size, ffm.num_of_field)
	n := len(ffm.conf.FeatureList)
	for i, feaInfo := range ffm.conf.FeatureList {
		info := fmt.Sprintf("%d:%d", feaInfo.SlotId, feaInfo.Cross)
		metaLine += info
		if i != n-1 {
			metaLine += "\t"
		}
	}
	err := ffm.model.save(path, metaLine)
	return err
}

func (ffm *FFMModel) Eval(p bool) {

}

//
func (ffm *FFMModel) Predict(inslist []*base.Instance) ([]base.Result, error) {
	n := len(inslist)
	res := make([]base.Result, n, n)
	for i, ins := range inslist {
		x, _ := ffm.predictz(ins, false)
		res[i] = base.Result{UserId: ins.UserId, Label: ins.Label, Score: x}
	}
	return res, nil
}

func (ffm *FFMModel) Train(inslist []*base.Instance) error {
	for _, ins := range inslist {
		ffm.train(ins)
	}
	return nil
}

func (ffm *FFMModel) predictz(ins *base.Instance, initial bool) (float32, [][]float32) {
	n := len(ins.Feas)
	param := make([]*base.Weight, n, n)

	// ffm_vec = [m1, m2, ..m_N], m1=[w1, w2, ...w_N], N=num_of_field
	ffm_vec := make([]float32, int(ffm.emb_size)*ffm.num_of_field*ffm.num_of_field)
	// fm part gradient
	grad_vec := make([][]float32, n, n)
	// fm_norm
	ffm_norm := make([]float32, int(ffm.emb_size)*ffm.num_of_field)
	// field start with 1, 2, 3....., convert to slice index = (field - 1)*emb_size for fm,
	// index = (field-1)*emb_size*num_of_field for ffm_sign
	for i := 0; i < n; i++ {
		// by copy value, not by copy pointer!
		slot := ins.Feas[i].Slot
		fea := ins.Feas[i].Fea
		text := ins.Feas[i].Text
		param[i] = ffm.model.getWeight(fea, slot, text, initial)

		field := ffm.slot_to_field[uint64(slot)]

		if field > 0 {
			ffm_vec = ffm.inplaceAddFFM(ffm_vec, param[i].VecW, int(field))
			ffm_norm = ffm.inplaceAddNorm(ffm_norm, param[i].VecW, int(field)) // self fm interaction term norm
		}
	}

	ffm_score := float32(0.0)
	fm_score := float32(0.0)
	for field_i := 0; field_i < ffm.num_of_field; field_i++ {
		s_fm := ffm.calcFmScore(field_i, ffm_vec, ffm_norm)
		fm_score += s_fm
		for field_j := field_i + 1; field_j < ffm.num_of_field; field_j++ {
			s_ffm := ffm.calcFfmScore(field_i, field_j, ffm_vec)
			ffm_score += s_ffm
		}
	}
	z := ffm.model.getWeight(0, 0, "", false).W
	for i := 0; i < n; i++ {
		z += param[i].W
		field := ffm.slot_to_field[uint64(ins.Feas[i].Slot)]
		if field > 0 {
			grad_vec[i] = ffm.calcGrad(int(field), ffm_vec, param[i].VecW)
		}
	}
	z += float32(ffm_score + fm_score/2.0)
	return base.Sigmoid32(z), grad_vec
}

func (ffm *FFMModel) train(ins *base.Instance) {
	//glog.V(5).Info(">>> [train] input ins: ", ins.String())
	p, ffmGrad := ffm.predictz(ins, true)
	curGrad := p
	label := ins.Label
	if label > 0 {
		curGrad -= 1.0
	}
	opt := ffm.optim
	m := len(ins.Feas)
	ffm.model.update(0, 0, label, curGrad, opt)
	for j := 0; j < m; j++ {
		key := ins.Feas[j].Fea
		slot := ins.Feas[j].Slot
		ffm.model.update(key, slot, label, curGrad, opt)
	}
	for j := 0; j < m; j++ {
		key := ins.Feas[j].Fea
		slot := ins.Feas[j].Slot
		field := int(ffm.slot_to_field[uint64(slot)])
		if field > 0 {
			gradVec := ffmGrad[j]
			for i := 0; i < int(ffm.full_size); i++ {
				gradVec[i] *= curGrad
			}
			ffm.model.updateEmb(key, slot, label, gradVec, opt)
		}
	}
}

func (ffm *FFMModel) buildGrad(field int, grad float32, ffmGrad []float32) []float32 {
	gradVec := make([]float32, ffm.full_size)

	return gradVec
}

func (ffm *FFMModel) inplaceAddFFM(vecFfm []float32, vecW []float32, field int) []float32 {
	size := int(ffm.emb_size) * ffm.num_of_field
	startIndex := size * (field - 1)
	for i := 0; i < size; i++ {
		vecFfm[startIndex+i] += vecW[i]
	}
	return vecFfm
}

func (ffm *FFMModel) inplaceAddNorm(vecNorm []float32, vecW []float32, field int) []float32 {
	startIndex := (field - 1) * int(ffm.emb_size)
	for i := 0; i < int(ffm.emb_size); i++ {
		vecNorm[startIndex+i] += vecW[startIndex+i] * vecW[startIndex+i]
	}
	return vecNorm
}

func (ffm *FFMModel) calcFmScore(field int, ffm_vec []float32, ffm_norm []float32) float32 {
	start_ffm_index := (field)*int(ffm.emb_size)*int(ffm.num_of_field) + (field)*int(ffm.emb_size)
	start_norm_index := (field) * int(ffm.emb_size)
	score := float32(0.0)
	//fmt.Println(">>>>>>", field)
	//fmt.Println("-----", start_ffm_index, start_norm_index)
	for i := 0; i < int(ffm.emb_size); i++ {
		w := (ffm_vec[start_ffm_index+i]*ffm_vec[start_ffm_index+i] - ffm_norm[start_norm_index+i])
		//fmt.Println(w)
		score += w
	}
	//fmt.Println("total score: ", field, score)
	return score
}

func (ffm *FFMModel) calcFfmScore(field_i int, field_j int, ffm_vec []float32) float32 {
	start_ffm_index_i := (field_i)*int(ffm.emb_size)*int(ffm.num_of_field) + (field_j)*int(ffm.emb_size)
	start_ffm_index_j := (field_j)*int(ffm.emb_size)*int(ffm.num_of_field) + (field_i)*int(ffm.emb_size)
	s := float32(0.0)
	for i := 0; i < int(ffm.emb_size); i++ {
		s += ffm_vec[start_ffm_index_i+i] * ffm_vec[start_ffm_index_j+i]
	}
	return s
}

func (ffm *FFMModel) calcGrad(field int, ffmVec []float32, vecW []float32) []float32 {
	field_i := field - 1
	gradVec := make([]float32, ffm.full_size)
	for field_j := 0; field_j < int(ffm.num_of_field); field_j++ {
		start_inner_index := field_j * int(ffm.emb_size)
		start_ffm_index := field_i*int(ffm.emb_size)*int(ffm.num_of_field) + start_inner_index
		// fm grad
		if field_j == field_i {
			for i := 0; i < int(ffm.emb_size); i++ {
				gradVec[start_inner_index+i] = ffmVec[start_ffm_index+i] - vecW[start_inner_index+i]
			}
		} else {
			// ffm grad
			daul_ffm_index := field_j*int(ffm.full_size) + field_i*int(ffm.emb_size)
			for i := 0; i < int(ffm.emb_size); i++ {
				gradVec[start_inner_index+i] = ffmVec[daul_ffm_index+i]
			}
		}
	}
	return gradVec
}
