package train_utils

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/golang/glog"

	"linearmodel/base"
	"linearmodel/dataloader"
	"linearmodel/metric"
	"linearmodel/model"
)

func CheckDir(dir string) ([]string, error) {
	ps, err := filepath.Glob(dir)
	if err != nil {
		return nil, err
	}
	if ps == nil {
		return nil, fmt.Errorf("No match file in %s", dir)
	}
	return ps, nil
}

func Test(p string) error {
	path := p
	f, err := os.Open(path)
	f.Close()
	if err != nil {
		return err
	}
	return nil
}

func ParsePath(s []string) ([]string, error) {
	rs := []string{}
	for _, si := range s {
		ps, err := CheckDir(si)
		if err != nil {
			return nil, err
		}
		for _, p := range ps {
			err := Test(p)
			if err != nil {
				return nil, err
			}
			rs = append(rs, p)
		}
	}
	return rs, nil
}

func run_test(test_list []string, m model.IModel, loader dataloader.IDataLoader, parallel int) (
	float64, float64, float64, float64, []base.Result) {
	pred := []base.Result{}
	for _, path := range test_list {
		pp := []base.Result{}
		pp = PredictParallel(m, loader, parallel, path)
		pred = append(pred, pp...)
	}
	//// metric
	if len(pred) == 0 {
		glog.Info("no test file")
		return 0.0, 0.0, 0.0, 0.0, nil
	}
	auc := metric.AUC(pred)
	gauc := metric.GroupAUC(pred)
	loss := metric.Losses(pred)
	return auc, loss, gauc, float64(len(pred)), pred
}

func TrainParallel(m model.IModel, loader dataloader.IDataLoader, parallel int, path string) {
	dataChan, err := loader.ReadFile(path)
	if err != nil {
		glog.Error("open file error: ", err)
		return
	}

	group := sync.WaitGroup{}
	for i := 0; i < parallel; i++ {
		group.Add(1)
		go func(i int) {
			defer group.Done()
			glog.V(3).Info("start worker: ", i)
			for p := range dataChan {
				err := m.Train(loader.ParseIns(p))
				if err != nil {
					glog.Error(err)
				}
			}
		}(i)
	}
	group.Wait()

}

func PredictParallel(m model.IModel, loader dataloader.IDataLoader, parallel int, path string) []base.Result {
	resChan := make(chan []base.Result, 2*parallel)
	dataChan, err := loader.ReadFile(path)
	if err != nil {
		glog.Error(err)
		return nil
	}
	group := sync.WaitGroup{}
	for i := 0; i < parallel; i++ {
		group.Add(1)
		go func() {
			defer group.Done()
			for p := range dataChan {
				x := loader.ParseIns(p)
				kr, _ := m.Predict(x)
				resChan <- kr
			}
		}()
	}
	go func() {
		group.Wait()
		close(resChan)
	}()
	res := []base.Result{}
	for r := range resChan {
		res = append(res, r...)
	}
	return res
}

func EvalParallel(m model.IModel, loader dataloader.IDataLoader, evalList []string, parallel int) {
	m.Eval(true)
	auc, loss, gauc, _, preds := run_test(evalList, m, loader, parallel)
	glog.Infof("test samples count=%d\nauc=%.5f\ngauc=%.5f\nloss=%.5f\n", len(preds), auc, gauc, loss)
}
