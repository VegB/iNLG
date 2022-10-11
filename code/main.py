import os
os.environ["DISABLE_TQDM"] = "1"
import os
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments
import wandb

import params
import utils 
from trainer import CLIPCapTrainer, ContraClipCapTrainer, SelfTrainer
from evaluate import text_evaluate, text_evaluate_gpt2


def _main(args):
    # Initialize directory
    args.current_output_dir = os.path.join(args.output_dir, f'seed-{args.random_seed}')
    utils.init_checkpoint_dir(args, args.current_output_dir)
    
    # Load Metrics, Scorers
    metric_list, scorers = utils.get_metrics_and_scores(args)

    # Prepare Model and Tokenizer
    model_name = utils.get_model_name(args)
    if args.mode in ['clipcap', 'contraclipcap']:
        config = args.lm_class.from_pretrained(args.lm_name).config
        if args.load_ckpt: 
            model = args.model_class.from_pretrained(model_name, config=config, args=args)
        else:
            model = args.model_class(config, args)
            model.lm = args.lm_class.from_pretrained(args.lm_name)
            if args.mode.find('clipcap') > -1 and not args.random_init_mapper: # init clip mapper weight from checkpoint
                model.clip_project = utils.load_model_ckpt(model.clip_project, args.clipcap_model_path, 'ClipCapMapper')
            if args.mode.find('clipcap') > -1 and not args.random_init_projection: # init clip mapper weight from checkpoint
                model.projection = utils.load_model_ckpt(model.projection, args.projection_model_path, 'TextProjection')
        model.assign_embedder()
    else:
        model = utils.load_lm(args, model_name)
    
    if args.save_mapper_ckpt:
        import torch
        torch.save(
            model.clip_project.state_dict(), 
            os.path.join(args.mapper_checkpoint_dir, f'{args.mapper_type}-{args.visual_prefix_length}_{args.dataset}-{args.model_type}_weights.pt'))
        exit(0)
    
    device = utils.get_device()
    model = model.to(device)
    tokenizer = utils.load_tokenizer(args, model_name)

    # Prepare Dataset
    raw_dataset = utils.load_dataset(args)
    processed_dataset = utils.encode_dataset(raw_dataset=raw_dataset, tokenizer=tokenizer, args=args)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Prepare Trainer
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = utils.postprocess_text(decoded_preds, decoded_labels)

        result = {}
        for metric, scorer in scorers.items():
            if metric == 'mauve':
                score = scorer.compute(predictions=decoded_preds, references=[l[0] for l in decoded_labels])
                result[metric] = score.mauve
            else:
                score = scorer.compute(predictions=decoded_preds, references=decoded_labels)
                result[metric] = ['score']
        return result

    trainer_args = Seq2SeqTrainingArguments(
        args.current_output_dir,
        num_train_epochs=args.train_epoch,
        do_eval=args.do_eval,
        evaluation_strategy='epoch' if args.do_eval else 'no',
        save_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=args.do_eval,
        metric_for_best_model=args.metric_for_best_model,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        label_smoothing_factor=args.label_smoothing_factor,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        predict_with_generate=True,
        fp16=False,
        disable_tqdm=1-args.verbose,
        remove_unused_columns=False
    )

    if args.dataset.find('with_image') > -1 and args.mode not in ['clipcap', 'contraclipcap']:
        print('Will remove "images" from validation set')
        for phase in ['validation', 'test']:
            raw_dataset[phase] = raw_dataset[phase].remove_columns(['images'])
            processed_dataset[phase] = processed_dataset[phase].remove_columns(['images'])
        
    if args.mode == 'clipcap':
        TrainerClass = CLIPCapTrainer
    elif args.mode == 'contraclipcap':
        TrainerClass = ContraClipCapTrainer
    else:
        TrainerClass = SelfTrainer  # Seq2SeqTrainer

    trainer = TrainerClass(
        model,
        trainer_args,
        additional_args=args,
        train_dataset=processed_dataset['train'],
        eval_dataset=processed_dataset['validation'],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # TRAINING / FINETUNING
    if not args.eval_only:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        print(f'Before evaluation, the best model checkpoint:\t{trainer.state.best_model_checkpoint}')
        if args.do_eval:
            args.val_step = utils.get_val_best_step(trainer.state.best_model_checkpoint)
        else:
            args.val_step = trainer.state.global_step

    # EVALUATION
    if args.task in ['concept2text', 'sentence_completion', 'image_captioning', 'story_generation']:
        for phase in args.test_set_list:
            if args.model_type == 'gpt2':
                text_evaluate_gpt2(
                    args=args,
                    model=model,
                    dataset=raw_dataset[phase],
                    tokenizer=tokenizer, 
                    phase=phase,
                    metric_list=metric_list,
                )
            else:
                text_evaluate(
                    args=args,
                    trainer=trainer,
                    dataset=processed_dataset[phase],
                    tokenizer=tokenizer, 
                    phase=phase,
                    metric_list=metric_list,
                )
    else:
        raise ValueError(f'Unknown task: {args.task}. Please specify one of ["concept2text", "concept2story", "dataset2text"].')


if __name__ == '__main__':
    # Load Args
    args = params.get_args()

    # Init wandb
    if args.allow_wandb:
        wandb.init(
            project='iNLG-public',
            tags=[args.mode, args.dataset, args.model_type],
            config=args,
        )

    for seed in args.random_seed_list:
        args.random_seed = seed
        _main(args)

    utils.compute_score_avg_std(args)
