# IJCAI Model
An implementation of our model with TensorFlow.

## Dataset

We provide FB15k and FB15k-237 datasets used in our experiment, the data set can be downloaded [here](https://1drv.ms/f/s!Aolzl9ayKKTzjjPHR_86EU6jUwR2)

Each Dataset is organized in the following format (using FB15k as example):

```reStructuredText
fb15k
 ├─train.txt: training data
 ├─valid.txt: valid data
 ├─test.txt: test data
 └─encode
    ├─entity_id.txt: entity to id mapping
    ├─relation_id.txt: relation to id mapping
    ├─train_encode.txt: encoded training data
    ├─valid_encode.txt: encoded valid data
    └─test_encode.txt:encoded test data
```

*dump.rdb* is a dump of Redis containing relational paths of each dataset. In the dump, *db0* and *db1* correspond to FB15k and FB15k-237 respectively. In each db, key is the composition of entity pair and number of steps, while value is a mapping from the relational path to its frequency in the form of JSON. For example, to query about the 2-step relational paths between entity 0 and entity 1020, the key is "0,1020:2", and the returned value is 

```json
{
  "[481, -767]": 1,
  "[285, -767]": 1,
  "[525, -767]": 1,
  "[84, -767]": 1
}
```



