import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
from typing import Tuple, Dict, List, Optional
import time
import os                                                                           
import numpy as np
from tqdm import tqdm
import collections
import math
import sys                                                                                                                                                                                                                                                                                                                                                                                                                                                                            

# 设置无缓冲输出
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

# -------------------------- 1. 经典CBOW数据集 --------------------------
class ClassicCBOWDataset(Dataset):
    def __init__(self, tokenized_path: str, word_to_idx: Dict[str, int], 
                 window_size: int = 5, max_samples: int = 1000000,
                 dataset_type: str = "train"):
        
        self.word_to_idx = word_to_idx
        self.vocab_size = len(word_to_idx)
        self.window_size = window_size
        self.dataset_type = dataset_type
        self.max_samples = max_samples
        
        # 特殊token
        self.pad_idx = 0  # 填充索引
        
        print(f"生成{dataset_type} CBOW样本...", flush=True)
        self.samples = self._generate_samples(tokenized_path)
        
        print(f"{dataset_type}数据集: {len(self.samples)} 样本, {self.vocab_size} 词汇", flush=True)
        
        # 统计信息
        self._analyze_dataset()

    def _generate_samples(self, tokenized_path: str):
        """生成CBOW样本"""
        df = pd.read_csv(tokenized_path, encoding="utf-8")
        samples = []
        
        pbar = tqdm(total=min(len(df), self.max_samples), desc=f"生成{self.dataset_type}样本")
        
        for tokens_str in df["tokenized_words"]:
            if len(samples) >= self.max_samples:
                break
                
            try:
                tokens = ast.literal_eval(tokens_str)
                sentence_samples = self._generate_sentence_samples(tokens)
                samples.extend(sentence_samples)
                pbar.update(1)
            except Exception as e:
                continue
        
        pbar.close()
        return samples[:self.max_samples]

    def _generate_sentence_samples(self, tokens: List[str]):
        """从单个句子生成CBOW样本"""
        sentence_samples = []
        n = len(tokens)
        
        for i in range(n):
            center_word = tokens[i]
            if center_word not in self.word_to_idx:
                continue
                
            center_idx = self.word_to_idx[center_word]
            
            # 收集上下文词索引
            context_indices = []
            start = max(0, i - self.window_size)
            end = min(n, i + self.window_size + 1)
            
            for j in range(start, end):
                if i != j and tokens[j] in self.word_to_idx:
                    context_indices.append(self.word_to_idx[tokens[j]])
            
            # 确保至少有一个上下文词
            if len(context_indices) > 0:
                sentence_samples.append((context_indices, center_idx))
        
        return sentence_samples

    def _analyze_dataset(self):
        """分析数据集统计信息"""
        if len(self.samples) == 0:
            return
            
        # 统计上下文长度
        context_lengths = [len(context) for context, _ in self.samples]
        avg_context_len = np.mean(context_lengths)
        
        # 统计中心词频率
        center_counts = collections.Counter()
        for _, center_idx in self.samples:
            center_counts[center_idx] += 1
        
        print(f"{self.dataset_type}数据集分析:", flush=True)
        print(f"  平均上下文长度: {avg_context_len:.2f}", flush=True)
        print(f"  最频繁中心词: {list(center_counts.most_common(5))}", flush=True)
        print(f"  样本数量: {len(self.samples)}", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        context_indices, center_idx = self.samples[idx]
        
        # 转换为tensor
        context_tensor = torch.tensor(context_indices, dtype=torch.long)
        center_tensor = torch.tensor(center_idx, dtype=torch.long)
        
        return context_tensor, center_tensor

    def get_vocab_info(self):
        return list(self.word_to_idx.keys()), self.word_to_idx

# -------------------------- 2. 动态填充的DataLoader --------------------------
def cbow_collate_fn(batch):
    """自定义collate函数处理变长上下文"""
    contexts, centers = zip(*batch)
    
    # 找到最大上下文长度
    max_len = max(len(ctx) for ctx in contexts)
    
    # 填充上下文
    padded_contexts = []
    for ctx in contexts:
        if len(ctx) < max_len:
            # 填充到最大长度
            padded = torch.cat([ctx, torch.zeros(max_len - len(ctx), dtype=torch.long)])
        else:
            padded = ctx
        padded_contexts.append(padded)
    
    # 堆叠
    contexts_tensor = torch.stack(padded_contexts)
    centers_tensor = torch.stack(centers)
    
    return contexts_tensor, centers_tensor

# -------------------------- 3. 经典CBOW模型 --------------------------
class ClassicCBOW(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 300, hidden_dim: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # 词嵌入层
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # CBOW网络
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),  # 使用Tanh激活函数
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        # 嵌入层初始化
        nn.init.uniform_(self.embeddings.weight, -0.1, 0.1)
        
        # 网络层初始化
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, context_indices):
        """
        前向传播
        Args:
            context_indices: [batch_size, context_length] 上下文词索引
        Returns:
            logits: [batch_size, vocab_size] 预测logits
        """
        # 获取上下文词嵌入 [batch_size, context_length, embedding_dim]
        context_embeds = self.embeddings(context_indices)
        
        # 计算掩码（处理填充）
        mask = (context_indices != 0).float()  # 假设0是填充索引
        
        # 平均池化上下文词向量 [batch_size, embedding_dim]
        # 使用掩码避免填充位置影响平均值
        sum_embeddings = (context_embeds * mask.unsqueeze(-1)).sum(dim=1)
        context_lengths = mask.sum(dim=1).unsqueeze(-1)  # [batch_size, 1]
        
        # 避免除以零
        context_lengths = torch.clamp(context_lengths, min=1)
        avg_context = sum_embeddings / context_lengths
        
        # 通过网络
        logits = self.network(avg_context)
        
        return logits
    
    def get_word_vectors(self):
        """获取词向量（嵌入层权重）"""
        return self.embeddings.weight.data

# -------------------------- 4. 改进的训练器 --------------------------
class ClassicCBOWTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()
        
        # 使用AdamW优化器
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=1e-3, 
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=10,  # 初始周期
            T_mult=2,  # 周期倍增因子
            eta_min=1e-5  # 最小学习率
        )
        
        self.best_val_acc = 0.0
        self.history = {
            'train_loss': [], 'train_acc': [], 
            'val_loss': [], 'val_acc': [],
            'learning_rate': []
        }

    def calculate_accuracy(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """计算准确率"""
        _, predicted = torch.max(outputs, dim=1)
        correct = (predicted == targets).sum().item()
        return correct / targets.size(0)

    def train_epoch(self, dataloader: DataLoader, epoch: int, total_epochs: int) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss, total_acc, total_samples = 0.0, 0.0, 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} 训练")
        
        for batch_idx, (contexts, centers) in enumerate(pbar):
            contexts, centers = contexts.to(self.device), centers.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(contexts)
            loss = self.loss_fn(outputs, centers)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()

            # 统计
            batch_size = centers.size(0)
            total_loss += loss.item() * batch_size
            acc = self.calculate_accuracy(outputs, centers)
            total_acc += acc * batch_size
            total_samples += batch_size

            # 更新进度条
            avg_loss = total_loss / total_samples
            avg_acc = total_acc / total_samples
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'avg_acc': f'{avg_acc:.4f}'
            })
        
        # 计算epoch平均值
        avg_loss = total_loss / total_samples
        avg_acc = total_acc / total_samples
        
        return avg_loss, avg_acc

    def evaluate(self, dataloader: DataLoader, epoch: int, total_epochs: int) -> Tuple[float, float]:
        """验证模型"""
        self.model.eval()
        total_loss, total_acc, total_samples = 0.0, 0.0, 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} 验证")
        
        with torch.no_grad():
            for contexts, centers in pbar:
                contexts, centers = contexts.to(self.device), centers.to(self.device)
                
                outputs = self.model(contexts)
                loss = self.loss_fn(outputs, centers)

                batch_size = centers.size(0)
                total_loss += loss.item() * batch_size
                acc = self.calculate_accuracy(outputs, centers)
                total_acc += acc * batch_size
                total_samples += batch_size

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{acc:.4f}'
                })
        
        avg_loss = total_loss / total_samples
        avg_acc = total_acc / total_samples
        
        return avg_loss, avg_acc

    def fit(self, train_dataloader: DataLoader, val_dataloader: DataLoader, 
            epochs: int = 20, save_dir: str = "/mnt/f/cbow_data/models"):
        """训练模型"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"开始经典CBOW训练 | 设备: {self.device}", flush=True)
        print(f"训练样本: {len(train_dataloader.dataset)} | 验证样本: {len(val_dataloader.dataset)}", flush=True)
        print(f"词汇表大小: {self.model.vocab_size}", flush=True)
        print(f"嵌入维度: {self.model.embedding_dim}", flush=True)
        
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            print(f"\n{'='*50}", flush=True)
            print(f"Epoch {epoch}/{epochs}", flush=True)
            print(f"{'='*50}", flush=True)
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_dataloader, epoch, epochs)
            
            # 验证
            val_loss, val_acc = self.evaluate(val_dataloader, epoch, epochs)
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)

            print(f"\nEpoch {epoch} 结果:", flush=True)
            print(f"训练 - 损失: {train_loss:.4f} | 准确率: {train_acc:.4f}", flush=True)
            print(f"验证 - 损失: {val_loss:.4f} | 准确率: {val_acc:.4f}", flush=True)
            print(f"学习率: {current_lr:.6f}", flush=True)

            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                model_path = os.path.join(save_dir, 'best_classic_cbow_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'vocab_size': self.model.vocab_size,
                    'embedding_dim': self.model.embedding_dim
                }, model_path)
                print(f"🔥 保存最佳模型 | 验证准确率: {val_acc:.4f}", flush=True)

            # 每5个epoch保存检查点
            if epoch % 5 == 0:
                checkpoint_path = os.path.join(save_dir, f'classic_cbow_checkpoint_epoch_{epoch}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'history': self.history,
                    'val_acc': val_acc,
                    'val_loss': val_loss
                }, checkpoint_path)
                print(f"💾 保存检查点: {checkpoint_path}", flush=True)

        total_time = time.time() - start_time
        print(f"\n{'='*50}", flush=True)
        print(f"训练完成! 总耗时: {total_time:.2f}s ({total_time/60:.2f}分钟)", flush=True)
        print(f"最佳验证准确率: {self.best_val_acc:.4f}", flush=True)
        print(f"{'='*50}", flush=True)
        
        # 保存最终模型
        final_model_path = os.path.join(save_dir, 'final_classic_cbow_model.pt')
        torch.save(self.model.state_dict(), final_model_path)
        print(f"保存最终模型: {final_model_path}", flush=True)
        
        return self.history

# -------------------------- 5. 词向量分析工具 --------------------------
def analyze_word_vectors(model, dataset, sample_words=None):
    """分析训练好的词向量"""
    print("\n" + "="*50, flush=True)
    print("词向量分析", flush=True)
    print("="*50, flush=True)
    
    # 获取词向量
    word_vectors = model.get_word_vectors()
    vocab, word_to_idx = dataset.get_vocab_info()
    
    print(f"词向量形状: {word_vectors.shape}", flush=True)
    
    # 计算相似度
    def cosine_similarity(vec1, vec2):
        return torch.nn.functional.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
    
    # 默认示例词
    if sample_words is None:
        sample_words = ["体育", "篮球", "足球", "电影", "音乐", "中国", "美国", "公司", "技术", "手机"]
    
    # 显示示例词的向量
    print("\n示例词向量:", flush=True)
    for word in sample_words:
        if word in word_to_idx:
            idx = word_to_idx[word]
            vector = word_vectors[idx]
            norm = torch.norm(vector).item()
            print(f"  '{word}': 范数={norm:.4f}", flush=True)
    
    # 计算相似词
    print("\n相似词分析:", flush=True)
    query_words = ["体育", "技术", "中国"]
    
    for query_word in query_words:
        if query_word not in word_to_idx:
            continue
            
        query_idx = word_to_idx[query_word]
        query_vec = word_vectors[query_idx]
        
        # 计算所有词的相似度
        similarities = []
        for i, word in enumerate(vocab):
            if i == query_idx or word == query_word:
                continue
            sim = cosine_similarity(query_vec, word_vectors[i])
            similarities.append((word, sim))
        
        # 取最相似的5个词
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_similar = similarities[:5]
        
        print(f"  '{query_word}' 的相似词:", flush=True)
        for word, sim in top_similar:
            print(f"    {word}: {sim:.4f}", flush=True)

# -------------------------- 6. 主函数 --------------------------
def main():
    # 参数配置
    args = {
        'epochs': 20,
        'batch_size': 256,
        'embedding_dim': 300,
        'hidden_dim': 512,
        'window_size': 5,
        'max_train_samples': 1000000,
        'max_val_samples': 100000,
        'save_dir': "/mnt/f/cbow_data/classic_models"
    }
    
    # 路径配置
    TRAIN_TOKEN_PATH = "data/thucnews/tokenized_data_bert.csv"
    TRAIN_CO_MATRIX_PATH = "data/thucnews/co_occurrence_matrix_torch.pt"
    VAL_TOKEN_PATH = "data/thucnews/tokenized_test_bert.csv"
    
    print("=" * 60, flush=True)
    print("经典CBOW词向量训练系统", flush=True)
    print("=" * 60, flush=True)
    print(f"训练轮数: {args['epochs']}", flush=True)
    print(f"批次大小: {args['batch_size']}", flush=True)
    print(f"嵌入维度: {args['embedding_dim']}", flush=True)
    print(f"隐藏层维度: {args['hidden_dim']}", flush=True)
    print(f"窗口大小: {args['window_size']}", flush=True)

    # 1. 加载统一的词汇表
    print("\n📁 加载词汇表...", flush=True)
    try:
        co_data = torch.load(TRAIN_CO_MATRIX_PATH)
        unified_word_to_idx = co_data["word_to_idx"]
        unified_vocab = co_data["vocab"]
        
        print(f"✅ 词汇表加载成功: {len(unified_vocab)} 个词", flush=True)
        
    except Exception as e:
        print(f"❌ 词汇表加载失败: {e}", flush=True)
        return

    # 2. 创建数据集
    print("\n📊 创建数据集...", flush=True)
    try:
        train_dataset = ClassicCBOWDataset(
            tokenized_path=TRAIN_TOKEN_PATH,
            word_to_idx=unified_word_to_idx,
            window_size=args['window_size'],
            max_samples=args['max_train_samples'],
            dataset_type="train"
        )
        
        val_dataset = ClassicCBOWDataset(
            tokenized_path=VAL_TOKEN_PATH,
            word_to_idx=unified_word_to_idx,
            window_size=args['window_size'],
            max_samples=args['max_val_samples'],
            dataset_type="val"
        )
        
        print(f"✅ 数据集创建成功", flush=True)
        print(f"   训练集: {len(train_dataset)} 样本", flush=True)
        print(f"   验证集: {len(val_dataset)} 样本", flush=True)
        
    except Exception as e:
        print(f"❌ 数据集创建失败: {e}", flush=True)
        return

    # 3. 创建数据加载器
    print("\n🔄 创建数据加载器...", flush=True)
    try:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=cbow_collate_fn
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=cbow_collate_fn
        )
        
        print(f"✅ 数据加载器创建成功", flush=True)
        print(f"   训练批次: {len(train_dataloader)}", flush=True)
        print(f"   验证批次: {len(val_dataloader)}", flush=True)
        
    except Exception as e:
        print(f"❌ 数据加载器创建失败: {e}", flush=True)
        return

    # 4. 创建模型
    print("\n🧠 创建模型...", flush=True)
    try:
        model = ClassicCBOW(
            vocab_size=len(unified_word_to_idx),
            embedding_dim=args['embedding_dim'],
            hidden_dim=args['hidden_dim']
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ 模型创建成功", flush=True)
        print(f"   参数总数: {total_params:,}", flush=True)
        print(f"   可训练参数: {trainable_params:,}", flush=True)
        print(f"   模型结构: 嵌入层({args['embedding_dim']}) → 隐藏层({args['hidden_dim']}) → 输出层({len(unified_word_to_idx)})", flush=True)
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}", flush=True)
        return

    # 5. 训练模型
    print("\n🚀 开始训练...", flush=True)
    try:
        trainer = ClassicCBOWTrainer(model)
        history = trainer.fit(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            epochs=args['epochs'],
            save_dir=args['save_dir']
        )
        
        print(f"✅ 训练完成!", flush=True)
        
    except Exception as e:
        print(f"❌ 训练失败: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return

    # 6. 分析词向量
    print("\n🔍 分析词向量...", flush=True)
    try:
        analyze_word_vectors(model, train_dataset)
    except Exception as e:
        print(f"词向量分析失败: {e}", flush=True)

    # 7. 保存训练历史
    print("\n💾 保存训练历史...", flush=True)
    try:
        history_df = pd.DataFrame(history)
        history_path = os.path.join(args['save_dir'], 'classic_cbow_training_history.csv')
        history_df.to_csv(history_path, index=False)
        print(f"✅ 训练历史已保存: {history_path}", flush=True)
        
        # 打印最终结果
        print(f"\n🎯 最终结果:", flush=True)
        print(f"   最佳验证准确率: {trainer.best_val_acc:.4f}", flush=True)
        if history['val_acc']:
            print(f"   最终验证准确率: {history['val_acc'][-1]:.4f}", flush=True)
        if history['train_acc']:
            print(f"   最终训练准确率: {history['train_acc'][-1]:.4f}", flush=True)
            
    except Exception as e:
        print(f"❌ 保存训练历史失败: {e}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("经典CBOW训练完成! 🎉", flush=True)
    print("=" * 60, flush=True)

if __name__ == "__main__":
    main()