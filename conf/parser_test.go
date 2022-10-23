package conf

import (
	"testing"
)

func TestConfigParser(t *testing.T) {
	path := "../test/test.conf"
	config := ParseConf(path)
	if config.OptimConfig == nil {
		t.Error("parse optimizer config error")
	}
	if config.OptimConfig.EmbSize != 12 {
		t.Error("parse optimizer config: embedding size error")
	}
	if len(config.FeatureList) != 2 {
		t.Errorf("parse feature len error: %d != %d", 2, len(config.FeatureList))
	}
	userFea := config.FeatureList[0]
	if userFea.SlotId != 101 || userFea.Name != "UserId" ||
		userFea.Cross != 1 || userFea.VecType != VectorType_LEFT {
		t.Error("parse user feature error")
	}
	itemFea := config.FeatureList[1]
	if itemFea.SlotId != 102 || itemFea.Name != "ItemId" ||
		itemFea.Cross != 2 || itemFea.VecType != VectorType_RIGHT {
		t.Error("parse item feature error")
	}
	trainList := config.TrainPathList
	if len(trainList) != 2 {
		t.Error("train list parse error")
	}
}
