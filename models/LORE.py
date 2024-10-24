import torch
import torch.nn as nn
from models.clip.prompt_learner import cfgc, cfgc_vitb32, load_clip_to_cpu, TextEncoder, PromptLearner_v2, PromptLearner_v4, PromptLearner_v3, PromptLearner_flower,  PromptLearner_nwpu,  PromptLearner_dog, PromptLearner_ucf
from utils.class_names import cifar10_classnames, cifar100_classnames, stanfordcars_classnames,  dtd_classnames, SAT_classnames, Aircraft_classnames, flower_classnames, nwpu_classnames, pattern_classnames,  imagenet_classnames, dog_classnames, ucf_classnames
from models.WB_module import White_box_module
import copy

class LORE(nn.Module):

    def __init__(self, args):
        super(LORE, self).__init__()
        if args["backbone"] == "vitb16":
            self.cfg = cfgc()
        elif args["backbone"] == "vitb32":
            self.cfg = cfgc_vitb32()
        clip_model = load_clip_to_cpu(self.cfg)
        self.clip_model = clip_model
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.class_num = 1
        self.top_k = args["top_k"]
        self.args = args
        self.layer_num = self.image_encoder.layers

        if args['dataset'] == 'cifar':
            self.class_num = 100
            self.prompt_learner = PromptLearner_v2(self.cfg, cifar100_classnames, self.clip_model)
        if args['dataset'] == 'cifar10':
            self.class_num = 10
            self.prompt_learner = PromptLearner_v2(self.cfg, cifar10_classnames, self.clip_model)
        if args['dataset'] == 'cars':
            self.class_num = len(stanfordcars_classnames)
            self.prompt_learner = PromptLearner_v2(self.cfg, stanfordcars_classnames, self.clip_model)
        if args['dataset'] == 'dtd':
            self.class_num = len(dtd_classnames)
            self.prompt_learner = PromptLearner_v2(self.cfg, dtd_classnames, self.clip_model)
        if args['dataset'] == 'sat':
            self.class_num = len(SAT_classnames)
            self.prompt_learner = PromptLearner_v3(self.cfg, SAT_classnames, self.clip_model)
        if args['dataset'] == 'aircraft':
            self.class_num = len(Aircraft_classnames)
            self.prompt_learner = PromptLearner_v4(self.cfg, Aircraft_classnames, self.clip_model)
        if args['dataset'] == 'flower':
            self.class_num = len(flower_classnames)
            self.prompt_learner = PromptLearner_flower(self.cfg, flower_classnames, self.clip_model)
        if args['dataset'] == 'nwpu':
            self.class_num = len(nwpu_classnames)
            self.prompt_learner = PromptLearner_nwpu(self.cfg, nwpu_classnames, self.clip_model)
        if args['dataset'] == 'pattern':
            self.class_num = len(pattern_classnames)
            self.prompt_learner = PromptLearner_nwpu(self.cfg, pattern_classnames, self.clip_model)
        if args['dataset'] == 'Imagenet':
            self.class_num = len(imagenet_classnames)
            self.prompt_learner = PromptLearner_v2(self.cfg, imagenet_classnames, self.clip_model)
        if args['dataset'] == 'dog':
            self.class_num = len(dog_classnames)
            self.prompt_learner = PromptLearner_dog(self.cfg, dog_classnames, self.clip_model)
        if args['dataset'] == 'ucf':
            self.class_num = len(ucf_classnames)
            self.prompt_learner = PromptLearner_ucf(self.cfg, ucf_classnames, self.clip_model)

        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.global_p = nn.Parameter(torch.randn(self.layer_num, self.args["prompt_length"], self.args["embd_dim"]))
        nn.init.normal_(self.global_p, std=0.02)

        self.classifier = nn.Linear(self.image_encoder.output_dim, self.class_num, bias=True).type(self.dtype)
        self.numtask = 0

        self.relu = nn.ReLU(inplace=True)
        self.bn1_ = nn.BatchNorm1d(self.text_encoder.text_projection.shape[1])
        self.bn2_ = nn.BatchNorm1d(self.text_encoder.text_projection.shape[1])
        self.linear_projection = copy.deepcopy(self.image_encoder.conv1)
        self.WB = White_box_module(use_stochastic=False, gcn_len=self.args["prompt_length_c"])


    def forward(self, image, target=None, p_target=None):
        logits = []
        class_features = self.image_encoder(image.type(self.dtype))
        class_features_ = class_features[:, 0, :]
        # class_features = class_features_

        prompts = self.prompt_learner().to(self.global_p.device)
        tokenized_prompts = self.tokenized_prompts.to(self.global_p.device)
        text_features = self.text_encoder(prompts, tokenized_prompts)  # class_num feature
        class_pool_key = text_features
        class_pool_key_norm = class_pool_key / class_pool_key.norm(dim=-1, keepdim=True)

        image_tokens = self.linear_projection(image.type(self.dtype))
        image_tokens = image_tokens.to(dtype=torch.float32)

        hidden_token, local_aware_p, att_tokens = self.WB(image_tokens, class_pool_key_norm.to(dtype=torch.float32))
        local_aware_p = local_aware_p.reshape(-1, self.layer_num, self.args["prompt_length_c"], self.args["embd_dim"]).type(self.dtype)
        local_aware_p = local_aware_p.permute(1, 0, 2, 3)

        image_features = self.image_encoder(image.type(self.dtype), self.global_p, local_aware_p, self.image_encoder.class_embedding)
        image_features = image_features[:, 0, :]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits.append(self.classifier(logit_scale * image_features))

        att_tokens_norm = att_tokens / att_tokens.norm(dim=-1, keepdim=True)
        hidden_token_norm = hidden_token / hidden_token.norm(dim=-1, keepdim=True)
        n = att_tokens_norm.shape[0]  # bs
        if target is not None:
            real_key_norm = att_tokens_norm[:, target, :].squeeze()
            target_index = torch.arange(n).to(self.global_p.device)
            target_index = target_index * (n + 1)
            real_key_norm = real_key_norm.reshape(n*n, -1)[target_index]
            s = real_key_norm * hidden_token_norm # B C
            increase_sim = torch.sum(s) / (real_key_norm.shape[0])
        else:
            increase_sim = logit_scale

        if p_target is not None:
            ind = torch.where(target != p_target)
            p_key_norm = att_tokens_norm[:, p_target, :].squeeze()
            p_target_index = torch.arange(n).to(self.global_p.device)
            p_target_index = p_target_index * (n + 1)
            p_key_norm = p_key_norm.reshape(n * n, -1)[p_target_index]
            p_key = p_key_norm[ind]
            s = p_key * hidden_token_norm[ind]
            reduce_sim = torch.sum(s) / (image_features.shape[0])
        else:
            reduce_sim = logit_scale

        return {
            'logits': torch.cat(logits, dim=1),
            'features': image_features,
            'increase_sim': increase_sim,
            'reduce_sim': reduce_sim,
        }

    def inference(self, image, target=None, p_target=None):
        logits = []
        image_tokens = self.linear_projection(image.type(self.dtype))
        image_tokens = image_tokens.to(dtype=torch.float32)
        _, local_aware_p = self.WB.inference(image_tokens)
        local_aware_p = local_aware_p.reshape(-1, self.layer_num, self.args["prompt_length_c"], self.args["embd_dim"]).type(self.dtype)
        local_aware_p = local_aware_p.permute(1, 0, 2, 3)
        image_features = self.image_encoder(image.type(self.dtype), self.global_p, local_aware_p, self.image_encoder.class_embedding)
        image_features = image_features[:, 0, :]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits.append(self.classifier(logit_scale * image_features))

        return {
            'logits': torch.cat(logits, dim=1),
            'features': image_features,
        }