
from transformers import AutoModelForCausalLM
from trl import SFTTrainer
import torch
from torch.nn import functional as F
from transformers import BitsAndBytesConfig
import os

class KLSFTTrainer(SFTTrainer):
    def __init__(self, *args, kl_coef=0.0, kl_ref_model_id=None, kl_ref_8bit=False,
                 kl_start_step=0, kl_t=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_coef = float(kl_coef)
        self.kl_ref_model_id = kl_ref_model_id
        self.kl_ref_8bit = bool(kl_ref_8bit)
        self.kl_start_step = int(kl_start_step)
        self.kl_t = float(kl_t)
        self._ref_model = None
    
    def _align_ref_vocab_(self, ref_model, student_model):
        """
        Ensure ref_model can ingest the student's input_ids:
        - Resize ref embeddings/lm head to student's vocab
        - Copy any *new* rows from student into ref to avoid random init
        """
        ref_emb = ref_model.get_input_embeddings()
        stu_emb = student_model.get_input_embeddings()
        ref_vocab = ref_emb.num_embeddings
        stu_vocab = stu_emb.num_embeddings

        if ref_vocab == stu_vocab:
            return  # nothing to do

        # 1) Resize ref
        ref_model.resize_token_embeddings(stu_vocab)
        # After resize, re-fetch (ties may be re-created)
        ref_emb = ref_model.get_input_embeddings()

        # 2) Copy newly added rows (if student has more)
        if stu_vocab > ref_vocab:
            with torch.no_grad():
                # copy input embeddings
                ref_emb.weight[ref_vocab:stu_vocab].copy_(
                    stu_emb.weight[ref_vocab:stu_vocab].to(ref_emb.weight.dtype)
                )

                # copy output head rows if lm_head exists and is not tied
                lm_head = getattr(ref_model, "lm_head", None)
                stu_head = getattr(student_model, "lm_head", None)
                if lm_head is not None and stu_head is not None:
                    # If weights were tied, resize_token_embeddings usually re-ties,
                    # but in case they're untied, mirror the new rows as well.
                    if lm_head.weight.shape[0] != ref_emb.weight.shape[0]:
                        # ensure lm_head is also resized by calling tie_weights
                        # (some architectures need an explicit tie)
                        try:
                            ref_model.tie_weights()
                        except Exception:
                            pass
                    if lm_head.weight.shape[0] == stu_head.weight.shape[0] == stu_vocab:
                        lm_head.weight[ref_vocab:stu_vocab].copy_(
                            stu_head.weight[ref_vocab:stu_vocab].to(lm_head.weight.dtype)
                        )

        # Good measure: re-tie after edits (no-op if already tied)
        try:
            ref_model.tie_weights()
        except Exception:
            pass

    def _lazy_init_ref(self):
        if self._ref_model is not None or self.kl_coef <= 0.0:
            return

        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True) if self.kl_ref_8bit else None

        # Load reference model
        ref_id = self.kl_ref_model_id
        self._ref_model = AutoModelForCausalLM.from_pretrained(
            ref_id,
            trust_remote_code=True,
            device_map={"": local_rank},
            torch_dtype=(None if self.kl_ref_8bit else torch.bfloat16),
            quantization_config=bnb_cfg,
        )

        # Align vocab with the *student* model (self.model) which is already wrapped by Trainer
        self._align_ref_vocab_(self._ref_model, self.model)

        # Freeze ref
        self._ref_model.eval()
        for p in self._ref_model.parameters():
            p.requires_grad_(False)


    def gem_loss(self, logits, labels, num_items_in_batch, beta=0.7, ignore_index=-100, h="logsigmoid"):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        mask = shift_labels != -100
        shift_logits = shift_logits[mask]
        shift_labels = shift_labels[mask]

        with torch.no_grad():
            logits_on_labels = torch.gather(
                shift_logits, dim=-1, index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)

            logits_diff = shift_logits - logits_on_labels.unsqueeze(-1)
            if h == "linear":
                weights = torch.ones_like(logits_diff)
            elif h == "logsigmoid":
                weights = F.sigmoid(0.01 * logits_diff)
            else:
                raise ValueError(h)

        gene_log_probs = F.log_softmax(shift_logits, dim=-1)
        q_probs = torch.exp(F.log_softmax(shift_logits / beta, dim=-1)).detach()

        real_log_probs = torch.gather(
            gene_log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        )

        if num_items_in_batch is not None:
            loss = -torch.sum(
                q_probs * weights * (real_log_probs - gene_log_probs), dim=-1
            ).sum() / num_items_in_batch
        else:
            loss = -torch.sum(
                q_probs * weights * (real_log_probs - gene_log_probs), dim=-1
            ).mean()

        return loss



    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Standard CE on assistant tokens (labels -100 elsewhere via TRL)
        labels = inputs.get("labels")
        if labels is None:
            raise RuntimeError(
                "Labels missing in inputs. Enable assistant_only_loss=True in SFTConfig so TRL builds the mask."
            )

        outputs = model(**inputs)
        logits = outputs.logits
        vocab = logits.size(-1)

        loss_ce = F.cross_entropy(
            logits.view(-1, vocab), labels.view(-1), ignore_index=-100
        )

        loss = loss_ce
        if self.kl_coef > 0.0 and (self.state.global_step or 0) >= self.kl_start_step:
            self._lazy_init_ref()
            with torch.no_grad():
                ref_out = self._ref_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                )
                ref_logits = ref_out.logits

            ref_logits = ref_logits.to(logits.dtype)
            mask = (labels != -100).to(logits.dtype)  # [B, T]
            # Forward KL: student || ref, temperature self.kl_t
            log_p = F.log_softmax(logits / self.kl_t, dim=-1)
            log_q = F.log_softmax(ref_logits / self.kl_t, dim=-1)
            p = log_p.exp()
            kl_tok = (p * (log_p - log_q)).sum(dim=-1)   # [B, T]
            denom = mask.sum().clamp_min(1.0)
            print(denom.sum(), )
            loss_kl = (kl_tok * mask).sum() / denom

            loss = loss + self.kl_coef * loss_kl

        return (loss, outputs) if return_outputs else loss
