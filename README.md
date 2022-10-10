# iNLG
This is the implementation code for "Visualize Before You Write: Imagination-Guided Open-Ended Text Generation".

## Environment Setup

### Step 1: Download repo
```
git clone git@github.com:VegB/iNLG.git
cd iNLG/
for dataset in activitynet commongen rocstories
do
    mkdir -p log/${dataset}
done
```


### Step 2: Setup conda environment
```
conda env create -f env.yml
conda activate inlg
python -m spacy download en
```


### Step 3: Download datasets & checkpoints
```
python scripts/download_data_and_checkpoint.py
```

## Text Generation

### For Concept2Text
```
# text-only
bash scripts/run_concept2text_text_only.sh

# with images
bash scripts/run_concept2text_with_image.sh
```

### For Sentence Completion
```
# text-only
bash scripts/run_sentence_completion_text_only.sh

# with images
bash scripts/run_sentence_completion_with_image.sh
```

### For Story Generation
```
# text-only
bash scripts/run_story_generation_text_only.sh

# with images
bash scripts/run_story_generation_with_image.sh
```