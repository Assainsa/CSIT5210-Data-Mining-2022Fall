## Requirements
```
python==3.9.13
recbole==1.0.1
cudatoolkit==11.3.1
pytorch==1.11.0
```

## Download Datasets and Pre-trained Model
The datasets and the pre-trained model from [Google Drive](https://drive.google.com/drive/folders/1Uik0fMk4oquV_bS9lXTZuExAYbIDkEMW?usp=sharing)

for the pretrain model, after unzipping, move `UniSRec-FHCKM-300.pth` to `saved/`

## Pre-train
Pre-train on one single GPU.
```
python pretrain.py
```

## Fine-tune
Fine-tune the pre-trained model in inductive setting.
```
python finetune.py -d Scientific -p saved/UniSRec-FHCKM-300.pth --train_stage=inductive_ft
```
replace `Scientific` to `Pantry`, `Instruments`, `Arts`, `Office` or `OR` to reproduce the results on paper.
replace inductive_ft to trasductive_ft to fintune with ID.


MOCO in `unisrec.py` and the SimCLR in `unisrec_ori.py`
