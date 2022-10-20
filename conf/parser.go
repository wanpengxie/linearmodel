package conf

import (
	"io/ioutil"

	"github.com/golang/glog"
	"github.com/golang/protobuf/proto"
)

func ParseConf(p string) *AllConfig {
	config := new(AllConfig)
	f, err := ioutil.ReadFile(p)
	if err != nil {
		glog.Fatalf("%v", err)
	}
	err = proto.UnmarshalText(string(f), config)
	if err != nil {
		glog.Fatalf("read config file %s error: %v", p, err)
	}

	// slot id should never be greater than 999
	for _, x := range config.FeatureList {
		if x.SlotId > 999 {
			glog.Fatalf("feature slot id greater than 999")
		}
	}
	return config
}
