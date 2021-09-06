# Deep Convolutional AutoEncoder
## Setup the project :
```bash
python3 -m venv /path/to/new/virtual/environment
source environment/bin/activate
pip install -r requirements.txt
pip install jupyter
ipython kernel install --name "local-venv" --user
```

## Train :
Modify the script train.sh with the different parameters you want.
```bash
./train.sh
```

## Use the model :
The notebook DeepAutoEncoder.ipynb contains examples on how to use the model either for reconstruction or clustering.


## Examples :

### Reconstruction :
![](test_reconstructions/../test%20reconstructions/test_4png.png)

### Clusters :
![](clustering_example.png)
