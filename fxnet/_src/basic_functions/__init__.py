MISSING_OPS = [
    # ---- math utilities ----
    "sqrt", "rsqrt", "square",
    "softplus", "sigmoid", "relu", "gelu", "gelu_tanh", "silu",
    "logsumexp", "softmax", "log_softmax",
    "norm", "var", "std",
    "argmax", "cumsum", "cumprod",

    # ---- indexing & masking ----
    "gather", "scatter", "index_select", "take_along_axis",
    "masked_fill", "masked_select",
    "tril", "triu",

    # ---- normalization ----
    "layer_norm", "rms_norm", "batch_norm", "group_norm", "weight_norm",

    # ---- convolution & pooling ----
    "conv1d", "conv2d", "conv3d", "conv_transpose2d",
    "avg_pool1d", "avg_pool2d", "avg_pool3d",
    "max_pool1d", "max_pool2d", "max_pool3d",
    "adaptive_avg_pool1d", "adaptive_avg_pool2d", "adaptive_avg_pool3d",
    "adaptive_max_pool1d", "adaptive_max_pool2d", "adaptive_max_pool3d",

    # ---- attention helpers ----
    "scaled_dot_product_attention",
    "einsum", "outer", "diagonal", "trace", "transpose",

    # ---- randomness & sampling ----
    "randn", "rand", "randint", "bernoulli",
    "normal", "multinomial", "topk", "dropout",

    # ---- losses ----
    "cross_entropy", "nll_loss", "mse_loss",
    "kl_div", "binary_cross_entropy_with_logits",

    # ---- optimizers (tensor-based) ----
    "sgd", "momentum", "adam", "adamw", "rmsprop", "adagrad",
    "clip_grad_norm", "clip_grad_value",

    # ---- embeddings & sequence helpers ----
    "embedding", "one_hot", "pad", "roll", "repeat_interleave",

    # ---- numerical stability ----
    "detach", "nan_to_num", "clamp_min", "clamp_max",
]
