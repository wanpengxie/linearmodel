package metric

import (
	"fmt"
	"math"
	"sort"

	"linearmodel/base"
)

type LabelFunc func(int) int

func AUC(result []base.Result) float64 {
	x, _ := calc_auc(result)
	return x
}

func GroupAUC(result []base.Result) float64 {
	resMap := make(map[uint64][]base.Result)
	for _, x := range result {
		resMap[x.UserId] = append(resMap[x.UserId], x)
	}
	count := 0.0
	sum := 0.0
	for _, v := range resMap {
		s := 0
		for _, x := range v {
			s += x.Label
		}
		if s == 0 || s == len(v) {
			continue
		}
		x, _ := calc_auc(v)
		sum += x * float64(len(v))
		count += float64(len(v))
	}
	// fmt.Println("group count: ", count)
	return sum / count
}

func Losses(result []base.Result) float64 {
	n := len(result)
	l := 0.0
	for i, _ := range result {
		l += Loss(result[i])
	}
	return l / float64(n)
}

func Loss(result base.Result) float64 {
	eps := 1e-9
	score := float64(result.Score)
	if score < eps {
		score = eps
	} else if score > 1.0-eps {
		score = 1.0 - eps
	}
	l := 0.0
	if result.Label > 0 {
		l = math.Log(score)
	} else {
		l = math.Log(1.0 - score)
	}
	return -l
}

func Mean(res []float64, weight []float64) (float64, float64) {
	mean := 0.0
	count := 0.0
	stdMean := 0.0
	resWeight := weight
	if resWeight == nil {
		for i := 0; i < len(res); i++ {
			resWeight = append(resWeight, 1.0)
		}
	}
	for i, x := range res {
		w := resWeight[i]
		mean += x * w
		count += w
	}
	for i, x := range res {
		w := resWeight[i]
		stdMean += (x - mean/count) * (x - mean/count) * w
	}
	return mean / count, math.Sqrt(stdMean / count)
}

func calc_auc(y []base.Result) (float64, error) {
	sort.Slice(y, func(i, j int) bool {
		return y[i].Score < y[j].Score
	})

	start, _, n := 0, 0, len(y)
	pcount, ncount, tmpcount := 0, 0, 0
	rankSum, rank := 0.0, 0
	i := 0
	for {
		if start > n-1 {
			break
		}
		if i == n || (i > start && y[i].Score != y[start].Score) {
			x := float64(tmpcount) * (float64(rank) + 0.5*float64(i-start+1))
			rankSum += x
			rank += (i - start)
			start = i
			tmpcount = 0
		}
		if i != n {
			if y[i].Label > 0 {
				pcount += 1
				tmpcount += 1
			} else {
				ncount += 1
			}
		}
		i++
	}
	auc := 0.0
	if pcount*ncount == 0 {
		return 0.0, fmt.Errorf("just one kind label: positive=%d, negtive=%d", pcount, ncount)
	}
	auc = (rankSum - 0.5*float64(pcount)*float64(pcount+1)) / float64(pcount) / float64(ncount)
	return auc, nil
}
