class Config:
    def __init__(
        self,
        vocab_size=50257,
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=3072,
        max_pos_embeddings=1024,
        attn_pdrop=0.1,
        embd_pdrop=0.1,
        resid_pdrop=0.1,
        layer_norm_epsilon=1e-5,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_pos_embeddings = max_pos_embeddings
        self.attn_pdrop = attn_pdrop
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon