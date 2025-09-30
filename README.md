This repository is for 'The Illusion of Forgetting: Post-hoc Utility Recovery from Unlearned Models'.

The reproduction pipeline is shown below. 

1. Prepare data. Split dataset into training and test sets.
```bash
cd ./scripts
bash data_pre.sh
```

2. Train Original models on different datasets. 
```bash 
cd ./scripts
bash train_org.sh
```

3. Conduct Unlearning on the targeted classes. 
```bash
cd ./scripts
bash train_unlearn.sh
```
4. Conduct our recovery procedure.
```bash
cd ./scripts
bash train_recovery.sh
```
