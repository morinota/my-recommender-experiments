from transformers import AutoTokenizer, AutoModel
import torch


class PretrainedLanguageModel:
    def __init__(self, model_name: str = "studio-ousia/luke-base") -> None:
        # モデルとトークナイザの初期化
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.language_model = AutoModel.from_pretrained(model_name)

    def forward(self, text_list: list[str]) -> torch.Tensor:
        """
        Args:
            text_list: list of texts
        Returns:
            embeddings: (batch_size, embedding_dim)
        """
        # トークナイズとテキストの処理
        inputs = self.tokenizer(text_list, padding=True, truncation=True, return_tensors="pt", max_length=512)
        # モデルによるテキストの埋め込み表現の取得
        with torch.no_grad():
            outputs = self.language_model(**inputs)

        # 最終的な埋め込みを取得(この例では[CLS]トークンをテキストの埋め込み表現とする)
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings


# 使用例
embedder = PretrainedLanguageModel()
texts = ["Hello, this is an example.", "Understanding and using LUKE embeddings.", "こんにちは"]
embeddings = embedder.forward(texts)
print(embeddings.shape)
print(embeddings)
