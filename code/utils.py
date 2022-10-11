import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
import datasets
import json
import h5py
import clip
import numpy as np
import torch
from collections import defaultdict
import wandb
import copy
from datasets import disable_caching
disable_caching()

import text_evaluation


gpt2_eot = '<|endoftext|>'

def format_concepts_concatenation(example, model):
    """
    For Concept2Text (CommonGen).
    Concatenate all the concepts into a string
    """
    if model in ['gpt2']:
        example['test_input_text'] = f'<|endoftext|> {", ".join(example["concepts"])}:'
        input_text = example['test_input_text'] + f' {example["target"]}' 
    else:
        input_text = ' '.join(example['concepts'])
    example['input_text'] = input_text
    return example


def format_sentence_completion_input_text(example, model):
    input_text = example['context']
    if model in ['gpt2']:
        example['test_input_text'] = example['context']
        input_text += f' {example["target"]}' 
    example['input_text'] = input_text
    return example


def format_image_captioning_input_text(example, model):
    input_text = ''
    if model in ['gpt2']:
        example['test_input_text'] = input_text
        input_text = f'{example["target"]}' 
    example['input_text'] = input_text
    return example


def format_story_generation_input_text(example, model):
    input_text = f'{example["title"]}. {example["context"]}'
    if model in ['gpt2']:
        example['test_input_text'] = input_text
        input_text += f' {example["target"]}' 
    example['input_text'] = input_text
    return example


def load_image_features(example, h):
    """Load pre-extracted image features from hdf5 file"""
    example['images'] = h[example['img_path']][()]
    return example


def load_dataset(args):
    """
    Load and process the data into desired format.
    CommonGen: Concatenate all the concepts into a list.
    """
    # datasets.set_progress_bar_enabled(args.verbose)  # progress bar
    if not args.verbose:
        datasets.logging.disable_progress_bar
    print(f'Loading dataset:\t {args.dataset}')

    base_dir = os.path.join(args.base_data_dir, args.task, args.dataset)
    data_files = {
        phase: os.path.join(base_dir, f'{phase}.jsonl') 
        for phase in args.phases
    }
    raw_dataset = datasets.load_dataset('json', data_files=data_files)

    # few-shot training data preparation
    if args.train_ratio < 1:
        num_all_train_data = len(raw_dataset['train'])
        num_train_data = int(num_all_train_data * args.train_ratio)
        raw_dataset['train'] = raw_dataset['train'].shuffle(seed=args.random_seed)
        if args.dataset.find('with_image') > -1:
            raw_dataset['train'] = raw_dataset['train'].filter(lambda example: example['img_path'] != '')
        raw_dataset['train'] = raw_dataset['train'].select(range(num_train_data))
        print(f'Few-shot training w/ {100*args.train_ratio:.2f}% of the training data\t'
                f'(sampled {len(raw_dataset["train"])} from {num_all_train_data})')

    # load image features
    if args.dataset.find('with_image') > -1:
        image_features_file = os.path.join(base_dir, f'{args.clip_features}.hdf5')
        print(f'Loading image features from {image_features_file}')
        with h5py.File(image_features_file, 'r') as h:
            processed_dataset = {}
            for phase, dataset in raw_dataset.items():
                dataset = dataset.map(load_image_features, fn_kwargs={'h': h})
                processed_dataset[phase] = dataset
            raw_dataset = processed_dataset

    # Preprocess functions (concatenate concepts into a string, etc)
    if args.task == 'concept2text':
        preprocess_func = format_concepts_concatenation
    elif args.task == 'sentence_completion':
        preprocess_func = format_sentence_completion_input_text
    elif args.task == 'image_captioning':
        preprocess_func = format_image_captioning_input_text
    elif args.task == 'story_generation':
        preprocess_func = format_story_generation_input_text

    processed_dataset = {}
    for phase, dataset in raw_dataset.items():
        dataset = dataset.map(preprocess_func, fn_kwargs={'model': args.model_type})
        if phase == 'train':  # no shuffle in dev/test
            processed_dataset[phase] = dataset.shuffle(seed=args.random_seed)
        else:
            processed_dataset[phase] = dataset

    return processed_dataset


def convert_to_features(example_batch, tokenizer, args):
    """
    Use the tokenizer to encode the input str.
    """
    if args.model_type in ['gpt2']:
        target_text = copy.deepcopy(example_batch['input_text'])
        if args.dataset.find('with_image') == -1:  # no visual input, add appending start-of-sentence token
            example_batch['input_text'] = [f'{gpt2_eot}{text}' for text in example_batch['input_text']]
    else:
        target_text = example_batch['target']
    input_encodings = tokenizer.batch_encode_plus(
        example_batch['input_text'], 
        pad_to_max_length=True,
        max_length=args.max_input_length,
    )
    target_encodings = tokenizer.batch_encode_plus(
        target_text, 
        pad_to_max_length=True,
        max_length=args.max_output_length,
    )
    features = input_encodings
    features['labels'] = target_encodings['input_ids'].copy()

    if args.mode.find('clipcap') > -1:
        # add prefix to the text, change the attention mask accordingly
        text_prefix = ''.join([tokenizer.pad_token] * args.visual_prefix_length)
        input_text_with_prefix = [f'{text_prefix}{text}' for text in example_batch['input_text']]
        tmp_encodings = tokenizer.batch_encode_plus(
            input_text_with_prefix, 
            pad_to_max_length=True,
            max_length=args.max_input_length + args.visual_prefix_length,
        )
        features['attention_mask'] = tmp_encodings['attention_mask'].copy()

    return features


def encode_dataset(raw_dataset, tokenizer, args):
    """
    Process the dataset, add encodings
    """
    # remove unused columns
    ignore_columns = ['input_text']
    if args.model_type in ['gpt2']:
        ignore_columns.append('test_input_text')
    elif args.task == 'concept2text':
        ignore_columns.append('concepts')
    elif args.task == 'sentence_completion':
        ignore_columns.extend(['context', 'label'])
    elif args.task == 'story_generation':
        ignore_columns.extend(['context', 'title'])
    if args.dataset.find('with_image') > -1:
        ignore_columns.append('img_path')
    
    processed_dataset = {}
    for phase, dataset in raw_dataset.items():
        # tokenize
        dataset = dataset.map(
            convert_to_features, 
            batched=True, 
            fn_kwargs={'tokenizer': tokenizer, 'args': args},
            load_from_cache_file=False,
        )
        # set the tensor type and remove unused columns
        columns = ['input_ids', 'attention_mask', 'labels']  # 'decoder_input_ids', 
        if args.dataset.find('with_image') > -1:
            columns.append('images')
        dataset.set_format(type='torch', columns=columns)
        processed_dataset[phase] = dataset.remove_columns(ignore_columns)   

    return processed_dataset


def get_model_name(args):
    """Get huggingface pretrained model name or local checkpoint"""
    if args.load_ckpt:
        # Specify Checkpoint
        args.resume_from_checkpoint = os.path.join(args.current_output_dir, f'checkpoint-{args.load_ckpt}')
        print(f'Will load checkpoint at:\t{args.resume_from_checkpoint}')
        return args.resume_from_checkpoint
    return args.lm_name


def load_lm(args, model_name):
    """
    Load and initialize the LM from the checkpoint.
    """
    model = args.lm_class.from_pretrained(model_name)
    print(f'Load Model from {model_name}')
    return model


def load_tokenizer(args, model_name):
    """
    Load and initialize the tokenizer from the checkpoint.
    """
    tokenizer = args.tokenizer_class.from_pretrained(model_name)
    if args.model_type == 'gpt2':
        tokenizer.pad_token = tokenizer.eos_token

    print(f'Load Tokenizer from {model_name}')
    return tokenizer

def _process_text(text):
    text = text.strip()
    if len(text) < 1:
        text = '<pad>'
    return text

def postprocess_text(preds, labels):
    preds = [_process_text(pred) for pred in preds]
    labels = [[_process_text(label)] for label in labels]
    return preds, labels


def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
        print(f'Create directory: {dir}')
    else:
        print(f'Exists: {dir}')


def get_val_best_step(best_model_checkpoint):
    return best_model_checkpoint.split('-')[-1]


def write_prediction(metrics_dict, inputs, predictions, references, args, phase):
    rst = {
        'metrics': metrics_dict,
        'inputs': inputs,
        'predictions': predictions,
        'references': references,
    }
    step_for_eval = args.val_step if not args.eval_only else args.load_ckpt
    prediction_name = f'{phase}-step{step_for_eval}'
    output_path = os.path.join(args.prediction_dir, f'{prediction_name}.json')
    with open(output_path, 'w') as fout:
        json.dump(rst, fout)
    print(f'Prediction for {phase}-set written to {output_path}')
    scores_path = os.path.join(args.score_dir, f'{phase}-seed-{args.random_seed}.json')
    with open(scores_path, 'w') as fout:
        json.dump(metrics_dict, fout, indent=4)
    print(f'Scores written to {scores_path}')


def get_metrics_and_scores(args):
    # Metric for NLGEval
    metric_list = [
        'Bleu_4', 'METEOR', 'CIDEr', 'SPICE', 'BERTScore', 
        'MAUVE', 'rep-1', 'rep-2', 'rep-3', 'rep-4', 'Diversity',
    ]
    if args.skip_spice:
        print('Will skip SPICE!')
        metric_list.remove('SPICE')
    if args.skip_meteor:
        print('Will skip METEOR!')
        metric_list.remove('METEOR')
    
    # Metrics for Trainer eval
    scorers = {}
    if args.do_eval and args.metric_for_best_model in ['sacrebleu', 'mauve']:
        scorers[args.metric_for_best_model] = datasets.load_metric(args.metric_for_best_model)
    return metric_list, scorers


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Device: {device}')
    return device


def load_model_ckpt(model, ckpt_path, model_name):
    model.load_state_dict(torch.load(ckpt_path))
    print(f'Load {model_name} ckpt from {ckpt_path}')
    return model


def load_clipcap_models(args, ClipCaptionModel):
    from transformers import GPT2Tokenizer
    # CLIP & GPT2
    clip_model, preprocess = clip.load("ViT-B/32", jit=False)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    # CLIPCap
    model_path = args.clipcap_model_path
    if not os.path.exists(model_path):
        raise ValueError(f'Can not find CLIPCap model at {model_path}')
    model = ClipCaptionModel(args.visual_prefix_length)
    model = load_model_ckpt(model, model_path, 'ClipCaptionModel')
    clipcap_model = model.eval()
    return clip_model, preprocess, tokenizer, clipcap_model


def save_tensor(tensor, name):
    tmp_dir = os.path.join(os.getcwd(), 'tmp')
    check_dir(tmp_dir)
    filename = os.path.join(tmp_dir, f'{name}.npy')
    print(f'Tensor shape: {tensor.size()}')
    array = tensor.cpu().detach().numpy()
    with open(filename, 'wb') as fout:
        np.save(fout, array)
    print(f'Saved to {filename}')


def norm(feat):
    new_feat = feat / feat.norm(dim=-1, keepdim=True)
    return new_feat


def compute_clip_similarity(feat1, feat2):
    cosine_similarity_matrix = feat1 @ feat2.T  # [batch_size, batch_size]
    similarity_scores = torch.diag(cosine_similarity_matrix, diagonal=0)  # get values on the diagonal
    return torch.clamp(similarity_scores, min=1e-13)


def linear_strech(val, l, h):
    return torch.clamp((val - l) / (h - l), min=1e-13)


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def get_dummy_token(batch_size: int, prefix_length: int, device: torch.device) -> torch.Tensor:
    return torch.zeros(batch_size, prefix_length, dtype=torch.int64, device=device)


def prepare_bartscore_input(rst_pred, rst_ref):
    """
    Inputs are in the format for pycocoeval.
    Outputs are prepared for BERTScore computing.
    """
    cand_list, refer_list = [], []
    for k in rst_pred:
        cand_list.append(rst_pred[k][0])
        refer_list.append(rst_ref[k])
    return cand_list, refer_list


def process_prediction_output(concept_list, pred_list, ref_list, remove_repeat=True):
    assert len(concept_list) == len(pred_list) == len(ref_list)
    print(f'#raw_pred = {len(pred_list)}\t#raw_gt = {len(ref_list)}')
    print(f'Will remove repeat predictions?\t{remove_repeat}')

    # for pycocoeval
    rst_pred = {}
    rst_ref = defaultdict(list)
    for idx, (concept, pred, ref) in enumerate(list(zip(concept_list, pred_list, ref_list))):
        key = ' '.join(sorted(concept.split())) if remove_repeat else idx
        rst_pred[key] = [pred]  # will remove repeating predictions for concept2text
        rst_ref[key].append(ref)

    # for bert-score
    cand_list, refer_list = prepare_bartscore_input(rst_pred, rst_ref)
    
    print(f'#raw pred = {len(pred_list)}\t#processed pred = {len(rst_pred)}')
    return rst_pred, rst_ref, cand_list, refer_list

def map_phase_name_for_wandb(phase):
    if phase == 'validation':
        return 'dev'
    return phase

def evaluate_prediction_output(concept_list, pred_list, ref_list, metric_list, args, phase):
    rst_pred, rst_ref, cand_list, refer_list = process_prediction_output(
        concept_list, pred_list, ref_list, remove_repeat=args.task=='concept2text')

    metrics_dict = text_evaluation.evaluator(gts=rst_ref, res=rst_pred, skip_spice=args.skip_spice, skip_meteor=args.skip_meteor)
    metrics_dict['BERTScore'] = text_evaluation.compute_bertscore(cand_list=cand_list, refer_list=refer_list)
    metrics_dict['MAUVE'] = text_evaluation.compute_mauve(pred_list=pred_list, ref_list=ref_list)
    repetition_scores = text_evaluation.compute_repetition(text_list=cand_list)
    for k, s in repetition_scores.items():
        metrics_dict[f'rep-{k}'] = s
    diversity_score = text_evaluation.compute_diversity(repetition_scores)
    metrics_dict['Diversity'] = diversity_score
    distinct_scores = text_evaluation.compute_distinct_n(text_list=cand_list)
    for k in range(1, 3):
        metrics_dict[f'distinct-{k}'] = distinct_scores[k]
    if args.allow_wandb:
        wandb.log({f'{map_phase_name_for_wandb(phase)}-{k}': v for k, v in metrics_dict.items()})

    print(metrics_dict)
    for m in metric_list:
        print(f'{m}:\t{100 * metrics_dict[m]:.2f}')
    
    write_prediction(metrics_dict, concept_list, pred_list, ref_list, args, phase)


def evaluate_data2text_prediction_output(input_list, pred_list, metric_list, args, phase):
    rst_pred = {str(i): [pred] for i, pred in enumerate(pred_list)}
    rst_ref = json.load(open(os.path.join(args.base_data_dir, args.task, args.dataset.split('_')[0], 'for_eval', f'{phase}_references.json')))
    cand_list, refer_list = prepare_bartscore_input(rst_pred, rst_ref)

    metrics_dict = text_evaluation.evaluator(gts=rst_ref, res=rst_pred, skip_spice=args.skip_spice)
    metrics_dict['BERTScore'] = text_evaluation.compute_bertscore(cand_list=cand_list, refer_list=refer_list)
    if args.allow_wandb:
        wandb.log({f'{map_phase_name_for_wandb(phase)}-{k}': v for k, v in metrics_dict.items()})

    print(metrics_dict)
    for m in metric_list:
        print(f'{m}:\t{100 * metrics_dict[m]:.2f}')
    
    write_prediction(metrics_dict, inputs=input_list, predictions=pred_list, references=[rst_ref[str(i)] for i in range(len(rst_ref))], args=args, phase=phase)


def dataset_to_list(dataset):
    num_data = len(dataset['concepts'])
    rtn = [{} for _ in range(num_data)]
    for i in range(num_data):
        for key in dataset.features:
            rtn[i][key] = dataset[key][i]
    return rtn


def init_checkpoint_dir(args, dir):
    args.prediction_dir = os.path.join(dir, 'predictions')
    check_dir(dir)
    check_dir(args.prediction_dir)


def compute_score_avg_std(args):
    for phase in args.test_set_list:
        metric_scores = defaultdict(list)
        for seed in args.random_seed_list:
            cur_score_file = os.path.join(args.score_dir, f'{phase}-seed-{seed}.json')
            scores = json.load(open(cur_score_file))
            for m, s in scores.items():
                metric_scores[m].append(s)
            # wandb.log({f'{map_phase_name_for_wandb(phase)}-{k}': v for k, v in scores.items()})
        score_statistics = {}
        for metric, score_list in metric_scores.items():
            score_statistics[metric] = {
                'mean': np.mean(score_list),
                'std': np.std(score_list),
            }
        if args.allow_wandb:
            wandb.log({f'{map_phase_name_for_wandb(phase)}-{k}-mean': v['mean'] for k, v in score_statistics.items()})
        score_statistics_file = os.path.join(args.score_dir, f'{phase}-all_scores.json')
        with open(score_statistics_file, 'w') as fout:
            json.dump(score_statistics, fout, indent=4)
        print(f'Score statistics written to {score_statistics_file}')
