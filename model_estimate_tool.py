from math import pow, log10


# by contrast,last time we trained a model for 7.87e+08 tokens with 2 epoches,
# which means total tokens trained are 1.57e+09 (4.79x smaller) and total
# consumed flops are 2.38e+18 (5.96x smaller)

def estimate(n_layer, n_dim, l_seq, vocab=32768):
    a, b = 0.5243, 0.4757
    M_base, D_base = 0.1715, 5.8316
    M = 72 * n_layer * pow(n_dim, 2) + 12 * n_layer * n_dim * l_seq
    n_1 = 12 * n_layer * pow(n_dim, 2)
    n_2 = n_1 + vocab * n_dim
    C = pow(M / M_base, 1 / a)
    D = D_base * pow(C, b)
    lr = 0.3118 * pow(C, -0.1250)
    batch_size = 0.2920 * pow(C, 0.3271)
    print(f"For model of n_layer:{n_layer},n_dim:{n_dim},l_seq:{l_seq}")
    print("=====")
    print(f"Estimated Model Size (N1)     : {n_1 / 1e6:.2f}M")
    print(f"Estimated Model Size (N2)     : {n_2 / 1e6:.2f}M")
    print(f"None Embedding FLOPs/Token(M) : {M / 1e6:.2f}M (10^{log10(M):.1f})")
    print(f"Computing Budget (C)          : {C:.2e} ")
    print(f"Data Scale (in tokens)        : {D:.2e} ")
    print(f"learning rate (lr)            : {lr:.2e}")
    print(f"batch size (in tokens)        : {batch_size:.2e}")
    print(f"batch size (n_samples)        : {batch_size / l_seq:.0f}")
    print(f"Estimate GPU Hours Consumed   : {C / 2.38e18 * 12:.1f} ")
    print('')


# previous model
estimate(24, 1024, 256)

estimate(16, 768, 256)
estimate(16, 768, 512)
