import torch
import torch.nn as nn

import transformers
from har_transformer.gpt2custom import GPT2Model



class HarTransformer (nn.Module):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            input_dim: int,
            labels_dim: int ,
            hidden_size: int = 256,
            max_length: int = None,
            transformers_layers: int = 12,
            n_positions: int = 1024,
            action_tanh: bool = True,
            **kwargs
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.labels_dim = labels_dim
        self.max_length = max_length
        self.action_tanh = action_tanh

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            n_layer= transformers_layers,
            n_positions=n_positions,
            n_ctx=n_positions,
            **kwargs
        )

        self.transformer = GPT2Model(config)

        self.embed_input = torch.nn.Linear(self.input_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        self.predict_label = nn.Sequential(
            *([nn.Linear(hidden_size, self.labels_dim)] + ([nn.Tanh()] if action_tanh else []))
        )

    def forward(self, inputs, attention_mask=None):

        batch_size, seq_length = inputs.shape[0], inputs.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long).to(inputs.device)
  
        input_embeddings = self.embed_input(inputs)



        stacked_inputs = self.embed_ln(input_embeddings).to(inputs.device)

        stacked_attention_mask = attention_mask
        stacked_attention_mask.to(dtype=torch.long, device=inputs.device)

        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        pred_labels = self.predict_label(x)
        return pred_labels

    def get_labels(self, inputs, **kwargs):
        # we don't care about the past rewards in this model

        inputs = inputs.reshape(1, -1, self.input_dim)


        if self.max_length is not None:
            inputs = inputs[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-inputs.shape[1]), torch.ones(inputs.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=inputs.device).reshape(1, -1)
            inputs = torch.cat(
                [torch.zeros((inputs.shape[0], self.max_length-inputs.shape[1], self.input_dim), device=inputs.device), inputs],
                dim=1).to(dtype=torch.float32)

        else:
            attention_mask = None

        label_pred = self.forward(inputs, attention_mask=attention_mask, **kwargs)

        return label_pred[0,-1]