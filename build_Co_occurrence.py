import pandas as pd
import torch
from typing import List,Dict,Tuple

def get_index(tokenized_data:List[List[str]]) -> Tuple[List[List],Dict[str,int]]:
    all_words = [word for tokens in tokenized_data for token in tokens]
    vocab = sorted(list(set(all_words)))
    word_to_idx = {
        word : idx for idx,
        word in enumerate(vocab)
    }
    print(f"词汇表大小: {len(vocab)}") 
    return vocab,word_to_idx

def create_co_occurrence_matrix_torch(
        tokenized_data:List[List[str]],
        window_size:int = 5,
        ignore_self: bool = True,
) -> Tuple[torch.Tensor,List[str],Dict[str,int]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"在 {device} 上构建共现矩阵")
    vocab,word_to_idx = get_index(tokenized_data)
    vocab_size = len(vocab)
    co_matrix = torch.zeros((vocab_size,vocab_size),dtype=torch.int32,device=device)
    half_window = (window_size - 1) // 2
    for tokens in tokenized_data:
        token_len = len(tokens)
        if token_len < 2:
            continue
        token_indices = torch.tensor(
            [word_to_idx[word] for word in tokens],
            dtype=torch.long,
            device=device
        )

        for i in range(token_len):
            current_idx = token_indices[i]
            start = max(0, i-half_window)
            end = min(token_len,i + half_window)
            window_indices = token_indices[start:end]
            if ignore_self:
                mask = torch.arange(len(window_indices)) != (i - start)
                window_indices = window_indices[mask]
                co_matrix[current_idx].scatter_add_(
                dim=0,  # 按第0维（列）累加
                index=window_indices,  # 要累加的列索引
                src=torch.ones_like(window_indices, dtype=torch.int32, device=device)  # 每次加1
            )
    return co_matrix,vocab,word_to_idx 

if __name__ == "__main__":  # 当脚本被直接运行时，才执行以下代码（导入时不执行）
    RESULT_PATH = "data/thucnews/tokenized_data_bert.csv"  # 文件路径
    result_df = pd.read_csv(RESULT_PATH, encoding="utf-8")  # pandas读CSV，指定编码（避免中文乱码）
    tokenized_data = [eval(tokens) for tokens in result_df["tokenized_words"].tolist()]
    co_matrix_dense, vocab, word_to_idx = create_co_occurrence_matrix_torch(
        tokenized_data,
        window_size=5,
        ignore_self=True
    )
    print("\n密集共现矩阵形状:", co_matrix_dense.shape)
    print("前10行10列预览:\n", co_matrix_dense[:10, :10].cpu())

    torch.save({
        "co_matrix": co_matrix_dense,  # 共现矩阵
        "vocab": vocab,                # 词汇表
        "word_to_idx": word_to_idx     # 词-索引映射
    }, "data/thucnews/co_occurrence_matrix_torch.pt")
    print("\nPyTorch共现矩阵已保存！")