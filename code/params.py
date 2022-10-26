import argparse
import os
from transformers import BartTokenizer, GPT2Tokenizer, T5Tokenizer, \
    BartForConditionalGeneration, T5ForConditionalGeneration, GPT2LMHeadModel

import utils
from model import ClipCaptionModel


LM_NAME = {
    'bart-base': 'facebook/bart-base',
    'bart-large': 'facebook/bart-large',
    't5-base': 't5-base',
    't5-large': 't5-large',
    'gpt2': 'gpt2',
    'gpt2-large': 'gpt2-large',
}

LM_CLASS = {
    'bart-base': BartForConditionalGeneration,
    'bart-large': BartForConditionalGeneration,
    't5-base': T5ForConditionalGeneration,
    't5-large': T5ForConditionalGeneration,
    'gpt2': GPT2LMHeadModel,
    'gpt2-large': GPT2LMHeadModel,
}

TOKENIZER_CLASS = {
    'bart-base': BartTokenizer,
    'bart-large': BartTokenizer,
    't5-base': T5Tokenizer,
    't5-large': T5Tokenizer,
    'gpt2': GPT2Tokenizer,
    'gpt2-large': GPT2Tokenizer,
}

MODEL_CLASS = {
    'clipcap': ClipCaptionModel,
    'contraclipcap': ClipCaptionModel,
}


def get_args():
    args_parser = argparse.ArgumentParser(description='Imagination-enhanced NLG')

    args_parser.add_argument('--base_data_dir', type=str, default='./data/')
    args_parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/')

    args_parser.add_argument('--task', type=str, default='concept2text', 
        choices=['concept2text', 'sentence_completion', 'image_captioning', 'story_generation'])
    args_parser.add_argument('--dataset', type=str, default='commongen')
    args_parser.add_argument('--mode', type=str, default=None, help='training mode')
    args_parser.add_argument('--model_type', type=str, help='the language model')
    args_parser.add_argument('--tune_lm', type=int, default=1)

    args_parser.add_argument('--train_ratio', type=float, default=1., help='reveal what portion of all training data')
    args_parser.add_argument('--train_epoch', type=int, default=1)
    args_parser.add_argument('--train_batch_size', type=int, default=64)
    args_parser.add_argument('--eval_batch_size', type=int, default=32)
    args_parser.add_argument('--max_input_length', type=int, default=32, help='max input length')
    args_parser.add_argument('--max_output_length', type=int, default=64, help='max decode length')

    args_parser.add_argument('--lr', type=float, default=2e-5)
    args_parser.add_argument('--weight_decay', type=float, default=0.01)
    args_parser.add_argument('--warmup_steps', type=int, default=0)
    args_parser.add_argument('--label_smoothing_factor', type=float, default=0.1)
    args_parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    args_parser.add_argument('--beam_search', type=int, default=1)
    args_parser.add_argument('--num_beams', type=int, default=10)
    args_parser.add_argument('--tok_p', type=float, default=0.9)
    args_parser.add_argument('--tok_k', type=int, default=0)
    args_parser.add_argument('--clip_features', type=str, default='vitb32')

    args_parser.add_argument('--test_set', type=str, default='test')
    args_parser.add_argument('--metric_for_best_model', type=str, default='sacrebleu')
    args_parser.add_argument('--load_ckpt', type=int, default=None, help='the epoch of the checkpoint')
    args_parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='the checkpoint to load')

    args_parser.add_argument('--eval_step_interval', type=int, default=500)
    args_parser.add_argument('--print_interval', type=int, default=50)
    args_parser.add_argument('--label', type=str, default=None)
    args_parser.add_argument('--allow_wandb', type=int, default=1)
    args_parser.add_argument('--verbose', type=int, default=0, help='whether to enable tqdm')
    args_parser.add_argument('--eval_only', action='store_true', help='skip training if true')
    args_parser.add_argument('--do_eval', action='store_true')
    args_parser.add_argument('--skip_spice', action='store_true', help='whether to skip spice evaluation')
    args_parser.add_argument('--skip_meteor', action='store_true', help='whether to skip meteor evaluation')
    args_parser.add_argument('--random_seeds', type=str, default='41,42,43')

    # Visual prefix
    args_parser.add_argument('--mapper_checkpoint_dir', type=str, default='./checkpoint/mapper/')
    args_parser.add_argument('--mapper_type', type=str, default='transformer')
    args_parser.add_argument('--random_init_mapper', type=int, default=0)
    args_parser.add_argument('--mapper_pretrain_dataset', type=str, default='coco-tunelm-wcontra')
    args_parser.add_argument('--visual_prefix_length', type=int, default=20)
    args_parser.add_argument('--visual_prefix_clip_length', type=int, default=10)
    args_parser.add_argument('--visual_prefix_dim', type=int, default=512)
    args_parser.add_argument('--tune_clip_mapper', type=int, default=0)
    args_parser.add_argument('--save_mapper_ckpt', action='store_true')

    # Text projection
    args_parser.add_argument('--projection_checkpoint_dir', type=str, default='./checkpoint/projection/')
    args_parser.add_argument('--random_init_projection', type=int, default=1)
    args_parser.add_argument('--projection_pretrain_dataset', type=str, default='coco')
    args_parser.add_argument('--tune_projection', type=int, default=1)

    # Contrastive
    args_parser.add_argument('--lambda_text_img', type=float, default=0.5)
    args_parser.add_argument('--lambda_text', type=float, default=0.5)
    args_parser.add_argument('--lambda_contrastive_loss', type=float, default=1.)
    args_parser.add_argument('--start_contrastive_at_epoch', type=int, default=10)

    args = args_parser.parse_args()

    # Model name, model class, tokenizer class
    try:
        args.lm_name = LM_NAME[args.model_type]
        args.lm_class = LM_CLASS[args.model_type]
        args.tokenizer_class = TOKENIZER_CLASS[args.model_type]
    except:
        raise ValueError(f'Model:\t{args.model_type}\tCurrently not supported.')
    if args.mode in ['clipcap', 'contraclipcap']:
        args.model_class = MODEL_CLASS[args.mode]

    # Dataset phases
    if args.task in ['sentence_completion']:
        args.phases = ['train', 'validation', 'test_both']
    elif args.task in ['image_captioning']:
        args.phases = ['train', 'validation', 'validation_full']
    else:
        args.phases = ['train', 'validation', 'test']
    args.test_set_list = args.test_set.split(',')

    args.output_dir = os.path.join(args.checkpoint_dir, f'{args.task}-{args.dataset}-{args.model_type}')
    if args.mode:
        args.output_dir = f'{args.output_dir}-{args.mode}'
    if args.label:
        args.output_dir = f'{args.output_dir}-{args.label}'
    args.score_dir = os.path.join(args.output_dir, 'scores')
    utils.check_dir(args.output_dir)
    utils.check_dir(args.score_dir)

    # prepare random seed
    args.random_seed_list = [int(seed) for seed in args.random_seeds.split(',')]

    args.clipcap_model_path = os.path.join(
        args.mapper_checkpoint_dir, 
        f'{args.mapper_type}-{args.visual_prefix_length}_{args.mapper_pretrain_dataset}-{args.model_type}_weights.pt')
    args.projection_model_path = os.path.join(
        args.projection_checkpoint_dir, 
        f'{args.projection_pretrain_dataset}-{args.model_type}_weights.pt')

    return args
