package model

import (
	"sync"
)

const CONNUM uint64 = 31

//type cmsCounter struct {
//	cms   *boom.CountMinSketch
//	mutex sync.Mutex
//}
//
//func (c *cmsCounter) count(key uint64, threshold int) bool {
//	bs := make([]byte, 8)
//	binary.LittleEndian.PutUint64(bs, key)
//	count := c.cms.Count(bs)
//	if int(count) > threshold {
//		return true
//	}
//	c.cms.Add(bs)
//	return false
//}

type concurrentCounter struct {
	counters sync.Map
}

func NewCounter() *concurrentCounter {
	c := &concurrentCounter{}
	c.Init()
	return c
}

func (c *concurrentCounter) Init() {

}

func (c *concurrentCounter) count(key uint64, threshold int) bool {
	val, _ := c.counters.Load(key)
	if val == nil {
		c.counters.Store(key, 1)
		return false
	}
	if val.(int) >= threshold {
		return true
	}
	c.counters.Store(key, val.(int)+1)
	return false
}
