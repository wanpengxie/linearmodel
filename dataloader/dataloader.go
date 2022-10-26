package dataloader

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"sync"

	"github.com/golang/glog"

	"linearmodel/base"
	"linearmodel/conf"
)

const LoaderBuffer = 200

var uidSlot = flag.Int("uid_slot", 101, "uid slot")
var fidSlot = flag.Int("fid_slot", 102, "fid slot")

type IDataLoader interface {
	Init(path string) error
	ReadFile(string) (<-chan []string, error)
	ParseIns([]string) []*base.Instance
	ParallelIterator(path string, worker int) <-chan []*base.Instance
}

type DataLoader struct {
	featureMap map[uint16]bool
	count      int
	dataChan   chan []string
	config     *conf.AllConfig
	readMu     sync.Mutex
	fileStatus bool
	isSigned   bool
}

func (b *DataLoader) Init(path string) error {
	b.featureMap = make(map[uint16]bool)
	conf := conf.ParseConf(path)
	b.config = conf
	b.isSigned = conf.IsFeatureSigned
	for _, x := range conf.FeatureList {
		b.featureMap[uint16(x.SlotId)] = true
	}
	return nil
}

func (b *DataLoader) readline(l string) *base.Instance {
	z := new(base.Instance)
	row := strings.SplitN(strings.TrimSuffix(l, "\n"), "\t", 2)
	if len(row) < 2 {
		glog.Errorf("line %s, format error: label not found", l)
		return nil
	}
	labelStr, feaListStr := row[0], row[1]
	label, err := strconv.Atoi(labelStr)
	if err != nil {
		glog.Errorf("parse label error: line=%s, label=%s, error=%v", l, labelStr, err)
		return nil
	}
	z.Label = label
	feaRow := strings.Split(feaListStr, " ")
	for _, str := range feaRow {
		slotStr, feaStr, found := strings.Cut(str, ":")
		if !found {
			glog.Errorf("feature field %s, format error, splitter not found", str)
			continue
		}
		slot, err := strconv.ParseUint(slotStr, 10, 64)
		if err != nil {
			glog.Errorf("parse feature slot=%s error: %v", slotStr, err)
			continue
		}
		if ok := b.featureMap[uint16(slot)]; !ok {
			continue
		}
		fea := uint64(0)
		text := ""
		if b.isSigned {
			fea, err = strconv.ParseUint(feaStr, 10, 64)
			if err != nil {
				glog.Errorf("parse feature sign=%s error: %v", feaStr, err)
				continue
			}
		} else {
			text = base.DeepCopyString(feaStr)
		}
		feature := base.Feature{Slot: uint16(slot), Fea: fea, Text: text}
		if !b.isSigned {
			feature.Encode()
		}
		if int(slot) == *uidSlot {
			z.UserId = feature.Fea
			z.UserIdStr = feature.Text
		}
		if int(slot) == *fidSlot {
			z.ItemId = feature.Fea
			z.ItemIdStr = feature.Text
		}
		z.Feas = append(z.Feas, &feature)
	}
	//if !b.isSigned {
	//	for i := range z.Feas {
	//		z.Feas[i].Encode()
	//	}
	//}
	return z
}

func (b *DataLoader) ReadFile(path string) (<-chan []string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}

	tryLock := b.readMu.TryLock()
	if !tryLock {
		return nil, fmt.Errorf("some other place is reading files %s", "")
	}
	b.dataChan = make(chan []string, 50)
	go func() {
		count := b.readFile(f)
		close(b.dataChan)
		f.Close()
		b.readMu.Unlock()
		glog.Infof("finish reading file: %s, count: %d", path, count)
	}()
	return b.dataChan, nil
}

func (b *DataLoader) readFile(f *os.File) int {
	r := bufio.NewReader(f)
	data := make([]string, 0, LoaderBuffer)
	count := 0
	i := 0
	for {
		l, e := r.ReadString('\n')
		if e != nil {
			if e == io.EOF && len(l) > 2 {
				data = append(data, l)
			}
			b.dataChan <- data
			break
		}
		if count == LoaderBuffer {
			i++
			b.dataChan <- data
			count = 0
			data = make([]string, 0, LoaderBuffer)
		}
		data = append(data, l)
		count++
	}
	return count
}

func (b *DataLoader) ParseIns(data []string) []*base.Instance {
	n := len(data)
	inslist := make([]*base.Instance, 0, n)
	for i := 0; i < n; i++ {
		ins := b.readline(data[i])
		if ins != nil {
			inslist = append(inslist, ins)
		}
	}
	return inslist
}

//func (b *DataLoader) GetData()  {
//	return b.dataChan
//}

func (b *DataLoader) ParallelIterator(path string, worker int) <-chan []*base.Instance {
	insChan := make(chan []*base.Instance, worker*2)
	dataChan, err := b.ReadFile(path)
	if err != nil {
		glog.Error(err)
		return nil
	}
	go func() {
		wg := sync.WaitGroup{}
		for i := 0; i < worker; i++ {
			wg.Add(1)
			go func(i int) {
				glog.V(3).Info("start parser: ", i)
				for data := range dataChan {
					ins := b.ParseIns(data)
					insChan <- ins
				}
				wg.Done()
				glog.V(3).Info("finish parser: ", i)
			}(i)
		}
		wg.Wait()
		close(insChan)
		glog.V(3).Info("finish all parsers")
	}()
	return insChan
}
