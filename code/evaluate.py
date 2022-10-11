import torch
from datasets import Dataset

import utils


def text_evaluate(args, trainer, dataset, tokenizer, metric_list, phase='val'):
    num_beams = args.num_beams if args.num_beams > 0 else None
    outputs = trainer.predict(
        test_dataset=dataset,
        max_length=args.max_output_length,
        num_beams=num_beams,
        metric_key_prefix=phase,
    )
    prediction_list = tokenizer.batch_decode(sequences=outputs.predictions, skip_special_tokens=True)

    input_list = tokenizer.batch_decode(sequences=dataset['input_ids'], skip_special_tokens=True)
    reference_list = tokenizer.batch_decode(sequences=outputs.label_ids, skip_special_tokens=True)
    utils.evaluate_prediction_output(input_list, prediction_list, reference_list, metric_list, args, phase)


def text_evaluate_gpt2(args, model, dataset, tokenizer, metric_list, phase='val'):
    if args.task in ['story_generation']:
        stop_token = tokenizer.eos_token
        period = '.'
        period_token_index = tokenizer.encode(period)[0]
    else:
        stop_token = "."
    stop_token_index = tokenizer.encode(stop_token)[0]
    prediction_list = []
    device = utils.get_device()

    for i, example in enumerate(dataset):
        tokens = None
        input_ids = tokenizer.encode_plus(text=example['test_input_text'], padding=False)['input_ids']
        input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device)
        if args.dataset.find('with_image') > -1:
            images = torch.Tensor(example['images']).unsqueeze(0).to(device)
            generated = model._concat_image_text_embeddings(images=images, input_ids=input_ids)
        else:
            model.embedder =  model.transformer.wte
            generated = model.embedder(input_ids)
        num_sentence = 0
        for _ in range(args.max_output_length):
            if args.dataset.find('with_image') > -1:
                outputs = model.lm(inputs_embeds=generated)
            else:
                outputs = model(inputs_embeds=generated)
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, -1).unsqueeze(1)
            next_token_emb = model.embedder(next_token)
            generated = torch.cat((generated, next_token_emb), dim=1)
            if tokens is None:
                tokens = next_token
            else:
                tokens = torch.cat((tokens, next_token), dim=1)
            if args.task in ['story_generation']:
                if period_token_index == next_token.item():
                    num_sentence += 1
                if num_sentence >= 4:
                    break
            if stop_token_index == next_token.item():
                break
        try:
            output_text = tokenizer.decode(list(tokens.squeeze().cpu().numpy()))
        except TypeError:
            output_text = ''
        if args.task in ['story_generation']:
            output_text = output_text.split(stop_token)[0]
        prediction_list.append(output_text)
        if not i % 100:
            print(f"[{i}]\tInput text:\t{example['test_input_text']}\n\tPrediction:\t{output_text}\n\tTarget:\t{example['target']}", flush=True)

    input_list = dataset['test_input_text']
    reference_list = dataset['target']
    utils.evaluate_prediction_output(input_list, prediction_list, reference_list, metric_list, args, phase)
