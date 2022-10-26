package model

import (
	"encoding/binary"
	"sync"

	boom "github.com/tylertreat/BoomFilters"
)

const CONNUM uint64 = 31

type cmsCounter struct {
	cms   *boom.CountMinSketch
	mutex sync.Mutex
}

func (c *cmsCounter) count(key uint64, threshold int) bool {
	bs := make([]byte, 8)
	binary.LittleEndian.PutUint64(bs, key)
	count := c.cms.Count(bs)
	if int(count) > threshold {
		return true
	}
	c.cms.Add(bs)
	return false
}

type concurrentCounter struct {
	counters [CONNUM]cmsCounter
}

func NewCounter() *concurrentCounter {
	c := &concurrentCounter{}
	c.Init()
	return c
}

func (c *concurrentCounter) Init() {
	for i := 0; i < int(CONNUM); i++ {
		c.counters[i].cms = boom.NewCountMinSketch(0.001, 0.99)
	}
}

func (c *concurrentCounter) count(key uint64, threshold int) bool {
	index := key % CONNUM
	counter := c.counters[index]
	counter.mutex.Lock()
	status := counter.count(key, threshold)
	counter.mutex.Unlock()
	return status
}
