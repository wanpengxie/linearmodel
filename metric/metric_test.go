package metric

import (
	"testing"

	"linearmodel/base"
)

func TestAUC(t *testing.T) {
	resList := []base.Result{{Label: 1, Score: 0.3}, {Label: 0, Score: 0.2}, {Label: 1, Score: 0.2}, {Label: 1, Score: 0.5},
		{Label: 0, Score: 0.8}}
	trueAuc := 0.41666667
	auc := AUC(resList)
	if base.NEQFloat(trueAuc, auc) {
		t.Errorf("auc = %.6f, true auc %.6f", auc, trueAuc)
	}
}

func TestGroupAUC(t *testing.T) {
	resList := []base.Result{{Label: 1, Score: 0.3, UserId: 10}, {Label: 0, Score: 0.2, UserId: 10},
		{Label: 1, Score: 0.7, UserId: 20}, {Label: 1, Score: 0.5, UserId: 20}, {Label: 0, Score: 0.6, UserId: 20},
		{Label: 1, Score: 0.2, UserId: 30}, {Label: 1, Score: 1.0, UserId: 30},
		{Label: 0, Score: 0.8, UserId: 40},
	}
	trueGauc := 0.7
	calcGauc := GroupAUC(resList)
	if base.NEQFloat(trueGauc, calcGauc) {
		t.Errorf("gauc = %.6f, true gauc = %.6f", calcGauc, trueGauc)
	}
}

func TestLosses(t *testing.T) {
	resList := []base.Result{{Label: 1, Score: 0.3}, {Label: 0, Score: 0.2}, {Label: 1, Score: 0.2}, {Label: 1, Score: 0.5},
		{Label: 0, Score: 0.8}}
	trueLoss := 1.06782787
	calcLoss := Losses(resList)
	if base.NEQFloat(trueLoss, calcLoss) {
		t.Errorf("calc loss = %.6f, true loss = %.6f", trueLoss, calcLoss)
	}
}
