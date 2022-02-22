This is the README.md file for WEAKLY SUPERVISED NUCLEI SEGMENTATION VIA INSTANCE LEARNING. 
Here we use MO dataset as an example to illustrate how to run our method.  

# Create environment

```angular2html
conda env create -f environment.yml 
```

# 1. Run SPN

```angular2html
python main.py --id SPN --cfg network/exp/MO/SPN.yaml --gpu 1
```

# 2. Run IEN

```angular2html
python main.py --id IEN --cfg network/exp/MO/IEN.yaml --gpu 1
```

# 3. Model inference

```angular2html
python main.py --id IEN_infer --cfg network/exp/MO/IEN_infer.yaml --gpu 1
```