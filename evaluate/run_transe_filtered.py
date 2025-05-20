
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

# 文件路径保持不变
triples_file = 'D:/........./test.tsv'  # 三元组 TSV 文件  
entity_mapping_file = 'D:/........./entities.tsv'         
relation_mapping_file = '/D:/........./DRKG/relations.tsv'          # 关系 ID 映射文件  
tails_file = 'D:/........./disease.tsv'                       # 替换的尾部实体 TSV 文件  
entity_embedding_file = 'D:/........./entity_embedding.npy'                                   # 实体嵌入文件  
relation_embedding_file = 'D:/........./relation_embedding.npy'                               # 关系嵌入文件  

def load_data_correctly():
    # 加载三元组
    triples = pd.read_csv(
        triples_file,
        sep='\t',
        header=None,
        names=['head', 'relation', 'tail'],
        dtype={'head': str, 'relation': str, 'tail': str}
    )

    # 加载实体映射
    entity_mapping = pd.read_csv(
        entity_mapping_file,
        sep='\t',
        header=None,
        names=['id', 'entity'],
        dtype={'id': 'int32', 'entity': str}
    ).set_index('entity')

    # 加载关系映射
    relation_mapping = pd.read_csv(
        relation_mapping_file,
        sep='\t',
        header=None,
        names=['id', 'relation'],
        dtype={'id': 'int32', 'relation': str}
    ).set_index('relation')

    # 加载 tails 文件（用于后续随机采样）
    tails_pool = pd.read_csv(
        tails_file,
        sep='\t',
        header=None,
        names=['id', 'new_tail'],
        dtype={'id': 'int32', 'new_tail': str}
    )

    # 加载嵌入
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    entity_embeddings = torch.from_numpy(np.load(entity_embedding_file)).float().to(device)
    relation_embeddings = torch.from_numpy(np.load(relation_embedding_file)).float().to(device)

    return triples, entity_mapping, relation_mapping, tails_pool, entity_embeddings, relation_embeddings, device


def vectorized_transE_l1_score(head_embs, relation_embs, tail_embs):
    """向量化计算TransE-L1得分（负的L1距离）"""
    # TransE公式：h + r - t 的L1范数，取负数使得得分越高表示结果越好
    return -torch.norm((head_embs + relation_embs) - tail_embs, p=1, dim=1)

def evaluate_triples(triples, entity_mapping, relation_mapping, entity_embeddings, relation_embeddings, tails_pool, device):
    rankings = []
    hits3 = 0
    hits10 = 0
    all_true_labels = []
    all_scores = []

    # 所有合法候选尾实体（确保能找到embedding）
    all_valid_tail_entities = tails_pool['new_tail'][tails_pool['new_tail'].isin(entity_mapping.index)].unique()

    for idx, triple in triples.iterrows():
        if (idx + 1) % 100 == 0:
            print(f"处理进度: {idx+1}/{len(triples)}")

        try:
            head_id = entity_mapping.loc[triple['head']]['id']
            relation_id = relation_mapping.loc[triple['relation']]['id']
            true_tail_id = entity_mapping.loc[triple['tail']]['id']
        except KeyError as e:
            print(f"跳过三元组 {triple}: 缺失实体或关系 {e}")
            continue

        # 生成包含真实尾实体的候选尾实体集合
        sampled_candidates = list(np.random.choice(all_valid_tail_entities, size=49, replace=False))
        if triple['tail'] not in sampled_candidates:
            sampled_candidates.append(triple['tail'])

        # 获取候选尾实体的ID和嵌入
        try:
            tail_ids = [entity_mapping.loc[tail]['id'] for tail in sampled_candidates]
        except KeyError as e:
            print(f"跳过三元组 {triple}: 候选尾实体映射失败 {e}")
            continue

        tail_embs = entity_embeddings[tail_ids].to(device)
        tail_ids_tensor = torch.tensor(tail_ids).to(device)

        head_emb = entity_embeddings[head_id].unsqueeze(0)
        relation_emb = relation_embeddings[relation_id].unsqueeze(0)

        with torch.no_grad():
            expanded_head = head_emb.expand(len(tail_ids), -1)
            expanded_rel = relation_emb.expand(len(tail_ids), -1)
            scores = vectorized_transE_l1_score(expanded_head, expanded_rel, tail_embs)

        # 计算真实尾实体在候选中的排名
        try:
            true_pos = (tail_ids_tensor == true_tail_id).nonzero().item()
        except:
            print(f"真实尾实体 {triple['tail']} (ID={true_tail_id}) 不在采样候选集中")
            continue

        sorted_indices = torch.argsort(scores, descending=True)
        true_rank = (sorted_indices == true_pos).nonzero().item() + 1
        print(true_rank)

        rankings.append(true_rank)
        hits3 += int(true_rank <= 3)
        hits10 += int(true_rank <= 10)

        true_score = scores[true_pos].item()
        false_scores = torch.cat([scores[:true_pos], scores[true_pos+1:]])
        all_true_labels.append(1)
        all_scores.append(true_score)
        all_true_labels.extend([0] * len(false_scores))
        all_scores.extend(false_scores.cpu().tolist())

    n_valid = len(rankings)
    if n_valid == 0:
        return np.nan, np.nan, 0.0, 0.0, 0.0

    MR = np.mean(rankings)
    MRR = np.mean(1.0 / np.array(rankings))
    HITS3 = hits3 / n_valid
    HITS10 = hits10 / n_valid
    AUC = roc_auc_score(all_true_labels, all_scores) if all_true_labels else 0.0

    return MR, MRR, HITS3, HITS10, AUC


def main():
    triples, entity_mapping, relation_mapping, replacement_tails, entity_embeddings, relation_embeddings, device = load_data_correctly()

    print(f"总三元组数: {len(triples)}")
    print(f"候选尾实体数: {len(replacement_tails)}")
    print(f"使用设备: {device}")

    mr, mrr, hits3, hits10, auc = evaluate_triples(
        triples, entity_mapping, relation_mapping,
        entity_embeddings, relation_embeddings,
        replacement_tails, device
    )

    print("\n评估结果:")
    print(f"MR: {mr:.2f}")
    print(f"MRR: {mrr:.4f}")
    print(f"HITS@3: {hits3:.4f}")
    print(f"HITS@10: {hits10:.4f}")
    print(f"AUC: {auc:.4f}")

if __name__ == "__main__":
    main()