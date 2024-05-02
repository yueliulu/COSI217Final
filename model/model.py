import torch.nn as nn
from transformers import BertModel

class BERTClass(nn.Module):
    def __init__(self, dr, num_class, hidden_dim):
        super(BERTClass, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
        self.dropout = nn.Dropout(dr)
        self.linear = nn.Linear(hidden_dim, num_class)

    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(
            input_ids,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids
        )
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output