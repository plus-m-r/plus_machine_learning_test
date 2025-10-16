import torch
from modelscope import AutoTokenizer, AutoModel
import pandas as pd

def get_bert_embeddings_as_tensor() -> list[torch.Tensor]:
    """
    从CSV文件中提取文本的BERT词嵌入，并转换为PyTorch Tensor格式
    
    返回:
        list: 嵌套列表结构，每个元素是一条文本的词嵌入Tensor，
              每个词嵌入是形状为(768,)的Tensor
    """
    model_name = "tiansz/bert-base-chinese"

    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # 读取数据
    df = pd.read_csv("data/thucnews/tokenized_data_bert.csv")
    data = df.iloc[:, -1]
    n = len(data)
    print(f"一共有{n}条数据")

    # 准备文本数据
    texts = [
        " ".join(tokens) for tokens in data
    ]

    # 分词处理
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        is_split_into_words=True
    )

    # 选择设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 获取模型输出
    with torch.no_grad():
        outputs = model(** inputs)

    token_embeddings = outputs.last_hidden_state
    all_word_embeddings = []

    # 处理每个句子
    for i in range(n):
        word_ids = inputs.word_ids(batch_index=i)
        word_embedding = []
        current_word_id = None
        current_word_vecs = []
        
        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            
            # 如果是新的词，计算上一个词的平均向量
            if word_id != current_word_id:
                if current_word_vecs:
                    # 对subword向量取平均
                    avg_vec = torch.mean(torch.stack(current_word_vecs), dim=0)
                    # 直接保持Tensor格式，无需转换为numpy
                    word_embedding.append(avg_vec.cpu())  # 移至CPU便于后续统一处理
                    current_word_vecs = []
                
                current_word_id = word_id
            
            # 添加当前subword的向量
            current_word_vecs.append(token_embeddings[i, idx, :])
        
        # 处理最后一个词
        if current_word_vecs:
            avg_vec = torch.mean(torch.stack(current_word_vecs), dim=0)
            word_embedding.append(avg_vec.cpu())
        
        # 将单条文本的词嵌入列表转换为Tensor（形状：[词数, 768]）
        all_word_embeddings.append(torch.stack(word_embedding))

    print(f"已处理完所有{len(all_word_embeddings)}条文本的词嵌入提取")
    print(f"第一条文本的词嵌入形状: {all_word_embeddings[0].shape}")  # 示例输出形状
    return all_word_embeddings

# 测试函数
if __name__ == "__main__":
    embeddings = get_bert_embeddings_as_tensor()
    print(f"返回结果类型: {type(embeddings)}")
    print(f"第一条文本的词嵌入类型: {type(embeddings[0])}")
