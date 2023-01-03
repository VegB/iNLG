from transformers import Seq2SeqTrainer
import clip
import torch
import torch.nn as nn
torch.autograd.set_detect_anomaly(True)
from transformers.trainer_pt_utils import get_parameter_names
from info_nce import InfoNCE
import wandb

import utils


class SelfTrainer(Seq2SeqTrainer):
    """Self-implemented Trainer."""

    def __init__(self, model=None, args=None, data_collator=None, train_dataset=None, eval_dataset=None, 
        tokenizer=None, model_init=None, compute_metrics=None, callbacks=None, optimizers=(None, None),
        additional_args=None):
        super().__init__(
            model=model, args=args, data_collator=data_collator, train_dataset=train_dataset,
            eval_dataset=eval_dataset, tokenizer=tokenizer, model_init=model_init, 
            compute_metrics=compute_metrics, callbacks=callbacks, optimizers=optimizers)

        self.additional_args = additional_args
        self.device = utils.get_device()
        self.counter = 0

    def compute_loss(self, model, inputs, return_outputs=False):
        # Original loss in Seq2SeqTrainer
        outputs = model(**inputs)
        loss = self.label_smoother(outputs, inputs['labels'])
        
        if not self.state.global_step % self.additional_args.print_interval:
            log_text = self._get_lm_log_text(inputs=inputs, outputs=outputs, lm_loss=loss)
            print(log_text, flush=True)
        if self.additional_args.allow_wandb:
            wandb.log({'lm_loss': loss})

        return (loss, outputs) if return_outputs else loss

    def _get_train_sampler(self):
        generator = torch.Generator()
        generator.manual_seed(42)
        return torch.utils.data.RandomSampler(self.train_dataset, generator=generator)

    def _get_lm_log_text(self, inputs, outputs, lm_loss):
        hyp_text = self.tokenizer.batch_decode(
            sequences=torch.argmax(outputs.logits, -1), skip_special_tokens=True)
        ref_text = self.tokenizer.batch_decode(sequences=inputs['labels'], skip_special_tokens=True)
        log_text = (f'[{self.state.global_step}]\tlm_loss:\t{lm_loss:.2f}\n'
                f'\tGT text:\t{ref_text[0]}\n\tGenerated text:\t{hyp_text[0]}')
        return log_text


class SelfTrainerWithClip(SelfTrainer):
    def __init__(self, model=None, args=None, data_collator=None, train_dataset=None, eval_dataset=None, 
        tokenizer=None, model_init=None, compute_metrics=None, callbacks=None, optimizers=(None, None),
        additional_args=None):
        super().__init__(
            model=model, args=args, data_collator=data_collator, train_dataset=train_dataset,
            eval_dataset=eval_dataset, tokenizer=tokenizer, model_init=model_init, 
            compute_metrics=compute_metrics, callbacks=callbacks, optimizers=optimizers,
            additional_args=additional_args)
        
        if self.additional_args.clip_features == 'vitb32':
            clip_model = 'ViT-B/32'
        elif self.additional_args.clip_features == 'rn50x4':
            clip_model = 'RN50x4'
        self.clip, _ = clip.load(clip_model, device=self.device)
        self.clip.eval()
        self.clip.to(self.device)

    def _get_reference_text_image_features(self, labels, image_features):
        ref_text = self.tokenizer.batch_decode(sequences=labels, skip_special_tokens=True)
        ref_text_inputs = clip.tokenize(ref_text, truncate=True).to(self.device)
        ref_text_features = utils.norm(self.clip.encode_text(ref_text_inputs))
        if len(image_features.size()) == 3:
            image_features = image_features[:, 0, :]
        ref_img_features = utils.norm(image_features).half()
        return ref_text, ref_text_features, ref_img_features
        

class CLIPCapTrainer(SelfTrainer):
    def create_optimizer(self):
        if self.optimizer is None:
            decay_parameters = []
            named_parameters = []
            # find tune-able parameters
            if self.additional_args.tune_lm:
                print('Will tune parameters in the LM')
                decay_parameters.extend(get_parameter_names(self.model.lm, forbidden_layer_types=[nn.LayerNorm]))
                named_parameters.extend(self.model.lm.named_parameters())
            if self.additional_args.tune_projection:
                print('Will tune parameters in the projection layer')
                decay_parameters.extend(get_parameter_names(self.model.projection, forbidden_layer_types=[nn.LayerNorm]))
                named_parameters.extend(self.model.projection.named_parameters())
            if self.additional_args.tune_clip_mapper:
                print('Will tune parameters in the CLIP mapper')
                decay_parameters.extend(get_parameter_names(self.model.clip_project, forbidden_layer_types=[nn.LayerNorm]))
                named_parameters.extend(self.model.clip_project.named_parameters())
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            print(f'#Tune-able parameters:\t{len(named_parameters)}')
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in named_parameters if n in decay_parameters],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in named_parameters if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer
 
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        lm_loss = self.label_smoother(outputs, inputs['labels'])

        if not self.state.global_step % self.additional_args.print_interval:
            log_text = self._get_lm_log_text(inputs=inputs, outputs=outputs, lm_loss=lm_loss)
            print(log_text, flush=True)
        if self.additional_args.allow_wandb:
            wandb.log({'lm_loss': lm_loss})
        
        return (lm_loss, outputs) if return_outputs else lm_loss

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.additional_args.max_output_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.additional_args.num_beams,
        }
        generated_tokens = model.generate(**inputs, **gen_kwargs)
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            with self.autocast_smart_context_manager():
                outputs = model(**inputs)
            loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
        if self.additional_args.allow_wandb:
            wandb.log({'eval_loss': loss})

        if self.args.prediction_loss_only:
            return (loss, None, None)

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return (loss, generated_tokens, labels)


class ContrastiveTrainer(SelfTrainerWithClip):
    def __init__(self, model=None, args=None, data_collator=None, train_dataset=None, eval_dataset=None, 
        tokenizer=None, model_init=None, compute_metrics=None, callbacks=None, optimizers=(None, None),
        additional_args=None):
        super().__init__(
            model=model, args=args, data_collator=data_collator, train_dataset=train_dataset,
            eval_dataset=eval_dataset, tokenizer=tokenizer, model_init=model_init, 
            compute_metrics=compute_metrics, callbacks=callbacks, optimizers=optimizers,
            additional_args=additional_args)
        
        self.contrastive_loss = InfoNCE()
        self.contrastive_started = False

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs['labels']
        ref_img_features = inputs['images']

        # Original loss in Seq2SeqTrainer
        outputs = model(**inputs)
        logits = outputs.logits
        lm_loss = self.label_smoother(outputs, labels)
        if self.additional_args.allow_wandb:
            wandb.log({'lm_loss': lm_loss})

        if self.state.epoch < self.additional_args.start_contrastive_at_epoch:
            loss = lm_loss
            if not self.state.global_step % self.additional_args.print_interval:
                log_text = self._get_lm_log_text(inputs=inputs, outputs=outputs, lm_loss=lm_loss)
            else:
                log_text = f'[{self.state.global_step}]\tloss:\t{loss:.2f}\tlm_loss:\t{lm_loss:.2f}'
        
        else:
            # Clear best state achieved before contarstive training starts
            if not self.contrastive_started:
                self.contrastive_started = True
                self.state.best_model_checkpoint = None 
                print(f'Contrastive learning start at epoch: {self.state.epoch}')
            
            # Contrastive loss
            # (1) generated text vs. generated image
            # (2) generated text vs. gt text
            hyp_text = self.tokenizer.batch_decode(
                sequences=torch.argmax(logits, -1), skip_special_tokens=True)
            hyp_text_features = outputs.projected_pred_feat
            ref_text, ref_text_features, ref_img_features = self._get_reference_text_image_features(
                labels=labels, image_features=ref_img_features)

            text_info_nce = self.contrastive_loss(
                query=hyp_text_features, positive_key=ref_text_features)
            text_img_info_nce = self.contrastive_loss(
                query=hyp_text_features, positive_key=ref_img_features)
            combined_info_nce = self.additional_args.lambda_text * text_info_nce + \
                (1-self.additional_args.lambda_text) * text_img_info_nce
            if self.additional_args.allow_wandb:
                wandb.log({'contra_loss': combined_info_nce})

            loss = lm_loss + self.additional_args.lambda_contrastive_loss * combined_info_nce

            log_text = (f'[{self.state.global_step}]\tloss:\t{loss:.2f}\tlm_loss:\t{lm_loss:.2f}'
                    f'\tcontrastive_loss:\t{combined_info_nce:.2f}\tlambda_text:\t{self.additional_args.lambda_text}'
                    f'\ttext_info_nce:\t{text_info_nce:.2f}\ttext_img_info_nce:\t{text_img_info_nce:.2f}\n'
                    f'\tGT text:\t{ref_text[0]}\n\tGenerated text:\t{hyp_text[0]}')
        
        if not self.state.global_step % self.additional_args.print_interval:
            print(log_text, flush=True)

        return (loss, outputs) if return_outputs else loss


class ContraClipCapTrainer(ContrastiveTrainer, CLIPCapTrainer):
    """
    Inherit compute_loss() from ContrastiveTrainer
    Inherit prediction_step() from CLIIPCapTrainer
    """
    pass
