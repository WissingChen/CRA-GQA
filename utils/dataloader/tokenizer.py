

def build_tokenizer(cfgs):
    if cfgs["model"]["lan"] == 'BERT':
        from transformers import BertTokenizer #DistilBertTokenizer
        tokenizer = BertTokenizer.from_pretrained(cfgs["model"]["lan_weight_path"])
    elif cfgs["model"]["lan"] == 'RoBERTa':
        from transformers import RobertaTokenizerFast,RobertaTokenizer
        tokenizer = RobertaTokenizerFast.from_pretrained(cfgs["model"]["lan_weight_path"])
    elif cfgs["model"]["lan"] == 'DeBERTa':
        from transformers import DebertaTokenizer
        tokenizer = DebertaTokenizer.from_pretrained(cfgs["model"]["lan_weight_path"])
    return tokenizer