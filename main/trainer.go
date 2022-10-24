package main

import (
	"flag"
	"time"

	"github.com/golang/glog"

	"linearmodel/conf"
	"linearmodel/dataloader"
	"linearmodel/model"
	"linearmodel/train_utils"
)

const NULL_STRING = "NULL"

var Parallel = flag.Int("parallel", 1, "parallel number")
var SaveDir = flag.String("save", NULL_STRING, "save directory")
var LoadDir = flag.String("load", NULL_STRING, "load directory")

//var IncSave = flag.String("save_inc", NULL_STRING, "save inc model")
//var IncLoad = flag.String("load_inc", NULL_STRING, "load inc model")
var conf_path = flag.String("conf", "", "config file path")
var stat = flag.Bool("stat", false, "model statistics")
var model_name = flag.String("model", "lr", "using model")

func main() {
	flag.Parse()

	save_path := *SaveDir
	load_path := *LoadDir
	//inc_load_path := *IncLoad
	//inc_save_path := *IncSave

	// config
	config := conf.ParseConf(*conf_path)
	glog.Infof("conf path: %s", *conf_path)

	train_list, _ := train_utils.ParsePath(config.TrainList)

	loader := new(dataloader.DataLoader)
	loader.Init(*conf_path)

	glog.Info(">>> initial loader sucess")

	var lm model.IModel
	if *model_name == "ffm" {
		lm = new(model.FFMModel)
	} else if *model_name == "lr" {
		lm = new(model.LRModel)
	} else if *model_name == "fm" {
		lm = new(model.FMModel)
	} else {
		glog.Fatalf("error model name: ", *model_name)
	}
	lm.Init(config)
	lm.Eval(false)
	glog.Info(">>> initial model sucess")

	glog.Infof("load model from %s\n", load_path)
	//glog.Infof("load inc model from %s\n", inc_load_path)
	glog.Infof("save model to %s\n", save_path)
	//glog.Infof("save inc model to %s\n", inc_save_path)
	if load_path != NULL_STRING {
		glog.Infof("=======load from model: %s=======", load_path)
		lm.Load(load_path)
	}

	t := time.Now()
	for _, path := range train_list {
		t := time.Now()
		train_utils.TrainParallel(lm, loader, *Parallel, path)
		glog.Infof("train %s time: [%s]\n", path, time.Now().Sub(t))
	}
	glog.Infof("train time: [%s]\n", time.Now().Sub(t))

	// ===================save model=========================
	if save_path != NULL_STRING {
		glog.Infof("=======save to model: %s=======", save_path)
		lm.Save(save_path)
	}

	// ====================eval list ========================
	train_utils.EvalParallel(lm, loader, config.PredictList, *Parallel)
	glog.Flush()

	// ====================predict list======================
	//predict_list := []string{}
	//for _, preds := range config.PredictList {
	//	l, _ := ParsePath(preds.TestPathList)
	//	predict_list = append(predict_list, l...)
	//}
}
