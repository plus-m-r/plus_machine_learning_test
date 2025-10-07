import pandas as pd
import os
import torch
from typing import List, Dict, Set, Callable, Optional
# 替换：导入 ModelScope 的 AutoTokenizer（而非 transformers 的 BertTokenizer）
from modelscope import AutoTokenizer  # 核心修改：使用 ModelScope 加载分词器

def load_raw_text(txt_dir: str, encoding: str = "utf-8") -> pd.DataFrame:
    """从文本文件加载原始数据并转换为DataFrame（逻辑不变）"""
    if not os.path.exists(txt_dir):
        raise FileNotFoundError(f"文件不存在: {txt_dir}")
    
    raw_data: List[Dict[str, str]] = []
    
    with open(txt_dir, "r", encoding=encoding, errors="ignore") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split("\t")
            if len(parts) == 2:
                category = parts[0].strip()
                original_text = parts[1].strip()
                raw_data.append({
                    "line_num": line_num,
                    "category": category,
                    "original_text": original_text
                })
    
    df = pd.DataFrame(raw_data)
    print(f"原始数据转DataFrame完成！共{len(df)}条有效数据")
    print(f"包含类别：{df['category'].unique().tolist()}")
    
    return df

def load_vocab(vocab_path: str) -> Set[str]:
    """加载词汇表文件（逻辑不变）"""
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"词汇表文件不存在: {vocab_path}")
    
    vocab: Set[str] = set()
    
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if word:
                vocab.add(word)
    
    print(f"词汇表加载完成，共包含{len(vocab)}个词")
    return vocab

def load_bert_tokenizer(model_name: str = "tiansz/bert-base-chinese") -> AutoTokenizer:
    try:
        # 关键修改：添加 use_fast=False，强制加载普通版 BertTokenizer（非 Rust 实现）
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            do_lower_case=False,
            use_fast=False  # 禁用 Fast 版，使用普通版分词器
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"ModelScope BERT分词器（普通版）加载完成，使用设备: {device}")
        print(f"分词器类型：{type(tokenizer).__name__}，词表大小：{len(tokenizer.get_vocab())}")
        return tokenizer
    except Exception as e:
        print(f"加载ModelScope BERT分词器失败: {str(e)}")
        raise

def create_bert_tokenizer(tokenizer: AutoTokenizer, vocab: Set[str], 
                         unk_token: str = "<UNK>") -> Callable[[str], List[str]]:
    """创建基于BERT的分词器函数（逻辑不变，兼容ModelScope Tokenizer）"""
    def tokenize_single(text: str) -> List[str]:
        # ModelScope Tokenizer 的 tokenize 方法与 transformers 完全一致
        tokens = tokenizer.tokenize(text)
        # 过滤不在自定义词汇表中的词，替换为未知词
        return [token if token in vocab else unk_token for token in tokens]
    
    return tokenize_single

def batch_tokenize_text(df: pd.DataFrame, tokenizer_func: Callable[[str], List[str]],
                       batch_size: int = 32) -> pd.DataFrame:
    """批量对DataFrame中的文本进行分词处理（逻辑不变）"""
    if "original_text" not in df.columns:
        raise ValueError("DataFrame必须包含'original_text'列")
    
    tokenized_results = []
    
    for i in range(0, len(df), batch_size):
        batch_texts = df["original_text"].iloc[i:i+batch_size].tolist()
        batch_results = [tokenizer_func(text) for text in batch_texts]
        tokenized_results.extend(batch_results)
        
        # 打印进度（每10批打印一次）
        if (i // batch_size) % 10 == 0:
            print(f"已处理 {min(i+batch_size, len(df))}/{len(df)} 条文本")
    
    df["tokenized_words"] = tokenized_results
    print(f"分词完成！已添加'tokenized_words'列到DataFrame")
    return df

def add_special_words_to_tokenizer(tokenizer: AutoTokenizer, vocab: Set[str]) -> None:
    """
    给 ModelScope 分词器添加自定义词汇表中的特殊词（逻辑不变，兼容ModelScope接口）
    """
    tokenizer_vocab = set(tokenizer.get_vocab().keys())
    missing_words = [word for word in vocab if word not in tokenizer_vocab]
    
    if missing_words:
        print(f"发现{len(missing_words)}个不在ModelScope分词器词表中的词汇，正在添加...")
        # ModelScope Tokenizer 支持 add_tokens 方法，与 transformers 一致
        tokenizer.add_tokens(missing_words)
        print(f"已成功添加{len(missing_words)}个新词到分词器，更新后词表大小：{len(tokenizer.get_vocab())}")
    else:
        print("自定义词汇表中的所有词已存在于ModelScope分词器词表中，无需添加")

if __name__ == "__main__":
    # 配置文件路径（需根据你的实际路径调整）
    RAW_TEXT_PATH = "data/thucnews/cnews.train.txt"    # 原始文本路径
    VOCAB_PATH = "data/thucnews/cnews.vocab.txt"      # 自定义词汇表路径
    OUTPUT_PATH = "data/thucnews/tokenized_data_bert.csv"  # 输出路径
    BERT_MODEL_NAME = "tiansz/bert-base-chinese"  
    
    try:
        # 1. 加载原始数据（逻辑不变）
        raw_df = load_raw_text(RAW_TEXT_PATH)
        
        # 2. 加载自定义词汇表（逻辑不变）
        vocab = load_vocab(VOCAB_PATH)
        
        # 3. 加载 ModelScope BERT 分词器（核心修改点）
        bert_tokenizer = load_bert_tokenizer(BERT_MODEL_NAME)
        
        # 4. 给分词器添加自定义词汇表中的特殊词（逻辑不变）
        add_special_words_to_tokenizer(bert_tokenizer, vocab)
        
        # 5. 创建分词函数（逻辑不变）
        tokenize_func = create_bert_tokenizer(bert_tokenizer, vocab)
        
        # 6. 批量分词处理（逻辑不变）
        result_df = batch_tokenize_text(raw_df, tokenize_func, batch_size=32)
        
        # 7. 预览结果（逻辑不变）
        print("\n最终DataFrame预览：")
        print(result_df[["category", "tokenized_words"]].head(2))
        
        # 8. 保存结果（逻辑不变）
        result_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
        print(f"\nBERT分词后的DataFrame已保存为：{OUTPUT_PATH}")
        
    except Exception as e:
        print(f"\n处理过程中发生错误: {str(e)}")