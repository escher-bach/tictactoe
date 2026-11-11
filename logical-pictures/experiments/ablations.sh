#!/bin/bash
# Which operations earn the result?  Each line learns a form with a restricted set of
# operations (1M chars for records, distant 200k for admission) and scores it on the test slice.
cd "$(dirname "$0")/.."
run() {  # name, extra args
  name=$1; shift
  python3 experiments/learn_text8.py --out runs/form_$name.pkl "$@" > runs/learn_$name.log 2>&1
  python3 experiments/eval_text8.py --form runs/form_$name.pkl --n 1000000 --tag $name --out runs/ablations.jsonl > runs/eval_$name.log 2>&1
}
run literal   --use S
run classes   --use S,C
run series    --use S,C,W
run recency   --use S,C,W,R,EQ
run pointers  --use S,C,W,R,EQ,PTR
run all       --use S,C,W,R,EQ,PTR,CLS,XC
run key1      --use S,C,W,R,EQ,PTR,CLS,XC --max-key 1
run key2      --use S,C,W,R,EQ,PTR,CLS,XC --max-key 2
run fine      --use S,C,W,R,EQ,PTR,CLS,XC --min-gain 0.0003 --max-sorts 48
