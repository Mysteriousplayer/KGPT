import torch
import torch.nn as nn

from models.clip import clip
from models.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.backbonename
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner_v2(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.NCTX  # number of context vectors
        dtype = clip_model.dtype
        device = clip_model.token_embedding.weight.device
        classnames = [f"This is a photo of a {c}" for c in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [name + "." for name in classnames]
        print(prompts)
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        with torch.no_grad():
            self.embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.CLASS_TOKEN_POSITION

    def forward(self):
        prompts = self.embedding
        return prompts

class PromptLearner_v4(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.NCTX  # number of context vectors
        dtype = clip_model.dtype
        device = clip_model.token_embedding.weight.device
        classnames = [f"a photo of a {c}" for c in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [name + ", a type of aircraft." for name in classnames]
        print(prompts)
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        with torch.no_grad():
            self.embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.CLASS_TOKEN_POSITION

    def forward(self):
        prompts = self.embedding
        return prompts

class PromptLearner_flower(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.NCTX  # number of context vectors
        dtype = clip_model.dtype
        device = clip_model.token_embedding.weight.device
        classnames = [f"a photo of a {c}" for c in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [name + ", a type of flower." for name in classnames]
        print(prompts)
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        with torch.no_grad():
            self.embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.CLASS_TOKEN_POSITION

    def forward(self):
        prompts = self.embedding
        return prompts


class PromptLearner_v3(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.NCTX  # number of context vectors
        dtype = clip_model.dtype
        device = clip_model.token_embedding.weight.device
        classnames = [f"a centered satellite photo of {c}" for c in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [name + "." for name in classnames]
        print(prompts)
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        with torch.no_grad():
            self.embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.CLASS_TOKEN_POSITION

    def forward(self):
        prompts = self.embedding
        return prompts


class PromptLearner_nwpu(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.NCTX  # number of context vectors
        ctx_init = cfg.CTXINIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        device = clip_model.token_embedding.weight.device
        classnames = [f"aerial imagery of a {c}" for c in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [name + "." for name in classnames]
        print(prompts)

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        with torch.no_grad():
            self.embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.CLASS_TOKEN_POSITION

    def forward(self):
        prompts = self.embedding
        return prompts

class PromptLearner_dog(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)

        n_ctx = cfg.NCTX  # number of context vectors
        ctx_init = cfg.CTXINIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        device = clip_model.token_embedding.weight.device
        classnames = [f"This is a photo of a {c}" for c in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [name + ", a type of dog." for name in classnames]
        print(prompts)
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        with torch.no_grad():
            self.embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.CLASS_TOKEN_POSITION

    def forward(self):
        prompts = self.embedding
        return prompts


class PromptLearner_ucf(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.NCTX  # number of context vectors
        ctx_init = cfg.CTXINIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        device = clip_model.token_embedding.weight.device
        classnames = [f"a photo of a person doing {c}" for c in classnames]
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [name + "." for name in classnames]
        print(prompts)
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        with torch.no_grad():
            self.embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.CLASS_TOKEN_POSITION

    def forward(self):
        prompts = self.embedding
        return prompts


class cfgc(object):
    backbonename = 'ViT-B/16'
    NCTX = 2
    CTXINIT = ''
    CSC = False
    CLASS_TOKEN_POSITION = 'end'

class cfgc_vitb32(object):
    backbonename = 'ViT-B/32'
    NCTX = 2
    CTXINIT = ''
    CSC = False
    CLASS_TOKEN_POSITION = 'end'