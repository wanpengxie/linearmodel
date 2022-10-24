# linearModel

LinearModel provide high-performance machine learning algorithoms written by golang. It contains logistic regression, factorization machine, field aware factorization machine. 

## Requirement
golang >= 1.18

## Usage
build
```shell
git clone https://github.com/wanpengxie/linearmodel.git linearmodel
cd linearmodel && go build -o trainer main/trainer.go    
```

train
```shell
# model_name in (lr, fm, ffm), parallel means thread numbers being used
./trainer -v=3 -logtostderr -conf ${conf_path} -parallel 10 -model ${model_name}   
```

save model to file
```shell
# save_path='/tmp/model'
./trainer -v=3 -logtostderr -conf ${conf_path} -parallel 10 -model ${model_name}  -save ${save_path}
```

load 
```shell
# load_path='/tmp/model'
./trainer -v=3 -logtostderr -conf ${conf_path} -parallel 10 -model ${model_name}  -load ${load_path}
```
## Config file format
We using protobuf as config file, like
```protobuf
optim_config {
  l1: 0.01
  l2: 0.02
  alpha: 2.0
  beta: 1.0
  emb_l1: 0.00
  emb_l2: 0.8
  emb_alpha: 0.8
  emb_beta: 1.0
  emb_size: 12
}

feature_list {
  name: "UserId"
  slot_id: 101
  vec_type: LEFT
  cross: 1
}

feature_list {
  name: "ItemId"
  slot_id: 102
  vec_type: RIGHT
  cross: 2
}

is_feature_signed: false
train_path_list:"path_to_train_data1"
train_path_list:"path_to_train_data2"
predict_path_list:"path_to_predict_data1"
predict_path_list:"path_to_predict_data2"
```

## Data file format
Using libsvm dataformat
, examples
```text
1   101:user1 102:item1 103:8 104:wednesday
0   101:user2 102:item2 103:9 104:wednesday
```


