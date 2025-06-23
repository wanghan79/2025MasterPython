import torch
from fvcore.nn import FlopCountAnalysis

from vit_model import Attention
from vit_model import vit_base_patch16_224_in21k as create_model


def main():
    # Self-Attention
    # a1 = Attention(dim=224, num_heads=1)
    # a1.proj = torch.nn.Identity()  # remove Wo
    #
    # # Multi-Head Attention
    # a2 = Attention(dim=224, num_heads=8)

    # vit_base_patch16_224_in21k
    model = create_model(num_classes=1000)

    # [batch_size, num_tokens, total_embed_dim]
    t = (torch.rand(1, 3, 224, 224),)
    #
    # flops1 = FlopCountAnalysis(a1, t)
    # print("Self-Attention FLOPs: {} M".format(flops1.total()/1000000))
    #
    # flops2 = FlopCountAnalysis(a2, t)
    # print("Multi-Head Attention FLOPs: {} M".format(flops2.total()/1000000))

    flops3 = FlopCountAnalysis(model, t)
    print("Multi-Head Attention FLOPs: {:.4f} G".format(flops3.total() / 1000000000))


if __name__ == '__main__':
    main()
