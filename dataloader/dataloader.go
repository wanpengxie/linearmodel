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
	ReadFile(string) error
	ParseIns([]string) []*base.Instance
	GetData() <-chan []string
	ParallelIterator(path string, worker int) <-chan []*base.Instance
}

type DataLoader struct {
	featureMap map[uint16]bool
	count      int
	dataChan   chan []string
	config     *conf.AllConfig
	readMu     sync.Mutex
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
	labelStr, feaListStr, found := strings.Cut(strings.TrimSuffix(l, "\n"), "\t")
	if !found {
		glog.Errorf("line %s, format error: label not found")
		return nil
	}
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
		if int(slot) == *uidSlot {
			z.UserId = fea
			z.UserIdStr = text
		}
		if int(slot) == *fidSlot {
			z.ItemId = fea
			z.ItemIdStr = text
		}
		z.Feas = append(z.Feas, base.Feature{Slot: uint16(slot), Fea: fea, Text: text})
	}
	if !b.isSigned {
		for i := range z.Feas {
			z.Feas[i].Encode()
		}
	}
	return z
}

func (b *DataLoader) ReadFile(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	tryLock := b.readMu.TryLock()
	if !tryLock {
		return fmt.Errorf("some other place is reading files")
	}
	b.dataChan = make(chan []string, 50)
	defer func() {
		close(b.dataChan)
		f.Close()
		b.readMu.Unlock()
	}()
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
	glog.V(3).Info("finish reading file: %s, line count: %d", path, count)
	return nil
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

func (b *DataLoader) GetData() <-chan []string {
	return b.dataChan
}

func (b *DataLoader) ParallelIterator(path string, worker int) <-chan []*base.Instance {
	insChan := make(chan []*base.Instance, worker*2)
	go func() {
		err := b.ReadFile(path)
		if err != nil {
			glog.Fatal(err)
		}
	}()
	dataChan := b.dataChan
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
