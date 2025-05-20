

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
from dataloader import *
from tqdm import tqdm
from torch.utils.data import DataLoader

class LLaDR(nn.Module):
    def __init__(self, base_model, nentity, nrelation, hidden_dim, gamma, 
                 double_entity_embedding, double_relation_embedding, triple_relation_embedding=False,
                 entity_text_embeddings=None, 
                 rho=0.4, zeta_2=0.3, zeta_3=0.5, distance_metric='cosine', 
                 text_dist_constraint="true", hake_p=0.5, hake_m=0.5, kwargs={}):
        
        super(LLaDR, self).__init__()
        self.model_name = base_model
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.distance_metric = distance_metric
        self.text_dist_constraint = text_dist_constraint
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        
        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        if double_relation_embedding:
            self.relation_dim = hidden_dim*2
        elif triple_relation_embedding:
            self.relation_dim = hidden_dim*3
        else:
            self.relation_dim = hidden_dim
        
        # Initialize relation embeddings
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        # Initialize randomly initialized component of entity embeddings
        self.entity_embedding_init = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding_init, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        ent_text_emb, ent_desc_emb = torch.chunk(entity_text_embeddings, 2, dim=1)
        
        # concatenate ent_text_emb[:self.entity_dim/2] and ent_desc_emb[:self.entity_dim/2]
        self.entity_text_embeddings = torch.cat([ent_text_emb[:, :self.entity_dim//2], ent_desc_emb[:, :self.entity_dim//2]], dim=1)
        self.entity_text_embeddings.requires_grad = False
        
        if base_model == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        
        if base_model not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE', 'HAKE']:
            raise ValueError('model %s not supported' % base_model)
            
        if base_model == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if base_model == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')
        
        # Hyperparameters
        self.rho = rho
        self.zeta_2 = zeta_2
        self.zeta_3 = zeta_3
        
        # model specific parameters
        # HAKE
        self.phase_weight = nn.Parameter(torch.Tensor([[hake_p * self.embedding_range.item()]]))
        self.modulus_weight = nn.Parameter(torch.Tensor([[hake_m]]))

    def get_entity_embedding(self):
        """
        Retrieve the embedding for the given entity ID.
        """
        return self.rho * self.entity_embedding_init + (1 - self.rho) * self.entity_text_embeddings

    def forward(self, sample, mode='single'):
        if mode == 'single':
            relation = torch.index_select(self.relation_embedding, dim=0, index=sample[:, 1]).unsqueeze(1)
            head_init = torch.index_select(self.entity_embedding_init, dim=0, index=sample[:, 0]).unsqueeze(1)
            tail_init = torch.index_select(self.entity_embedding_init, dim=0, index=sample[:, 2]).unsqueeze(1)
            head_text = torch.index_select(self.entity_text_embeddings, dim=0, index=sample[:, 0]).unsqueeze(1)
            tail_text = torch.index_select(self.entity_text_embeddings, dim=0, index=sample[:, 2]).unsqueeze(1)
            
            head_combined = self.rho * head_init + (1 - self.rho) * head_text
            tail_combined = self.rho * tail_init + (1 - self.rho) * tail_text
            
            link_pred_score = self.score_func(head_combined, relation, tail_combined, mode)
            
            return link_pred_score
            
        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            relation  = torch.index_select(self.relation_embedding, dim=0, index=tail_part[:, 1]).unsqueeze(1)
            tail_init = torch.index_select(self.entity_embedding_init, dim=0, index=tail_part[:, 2]).unsqueeze(1)
            head_init = torch.index_select(self.entity_embedding_init, dim=0, index=head_part.view(-1)).view(batch_size, negative_sample_size, -1)
            tail_text = torch.index_select(self.entity_text_embeddings, dim=0, index=tail_part[:, 2]).unsqueeze(1)
            head_text = torch.index_select(self.entity_text_embeddings, dim=0, index=head_part.view(-1)).view(batch_size, negative_sample_size, -1)
            
            tail_combined = self.rho * tail_init + (1 - self.rho) * tail_text
            head_combined = self.rho * head_init + (1 - self.rho) * head_text
            
            if self.text_dist_constraint == "true":
                text_dist = self.distance(tail_combined, tail_text)
            else:
                text_dist = torch.zeros((batch_size, 1), dtype=tail_combined.dtype, device=tail_combined.device)
            
            link_pred_score = self.score_func(head_combined, relation, tail_combined, mode)
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            relation  = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 1]).unsqueeze(1)
            head_init = torch.index_select(self.entity_embedding_init, dim=0, index=head_part[:, 0]).unsqueeze(1)
            tail_init = torch.index_select(self.entity_embedding_init, dim=0, index=tail_part.view(-1)).view(batch_size, negative_sample_size, -1)
            head_text = torch.index_select(self.entity_text_embeddings, dim=0, index=head_part[:, 0]).unsqueeze(1)
            tail_text = torch.index_select(self.entity_text_embeddings, dim=0, index=tail_part.view(-1)).view(batch_size, negative_sample_size, -1)
            
            head_combined = self.rho * head_init + (1 - self.rho) * head_text 
            tail_combined = self.rho * tail_init + (1 - self.rho) * tail_text 
            
            if self.text_dist_constraint == "true":
                text_dist = self.distance(head_combined, head_text)
            else:
                text_dist = torch.zeros((batch_size, 1), dtype=head_combined.dtype, device=head_combined.device)
            
            link_pred_score = self.score_func(head_combined, relation, tail_combined, mode)
        
        else:
            raise ValueError('mode %s not supported' % mode)
        
        return text_dist, link_pred_score 

    def distance(self, embeddings1, embeddings2):
        """
        Compute the distance between two sets of embeddings.
        """
        if self.distance_metric == 'euclidean':
            return torch.norm(embeddings1 - embeddings2, p=2, dim=-1)
        elif self.distance_metric == 'manhattan':
            return torch.norm(embeddings1 - embeddings2, p=1, dim=-1)
        elif self.distance_metric == 'cosine':
            embeddings1_norm = F.normalize(embeddings1, p=2, dim=-1)
            embeddings2_norm = F.normalize(embeddings2, p=2, dim=-1)
            cosine_similarity = torch.sum(embeddings1_norm * embeddings2_norm, dim=-1)
            cosine_distance = 1 - cosine_similarity
            return cosine_distance
        elif self.distance_metric == 'rotate':
            return self.rotate_distance(embeddings1, embeddings2)
        elif self.distance_metric == 'pi':
            pi = 3.14159262358979323846
            phase1 = embeddings1 / (self.embedding_range.item() / pi)
            phase2 = embeddings2 / (self.embedding_range.item() / pi)
            distance = torch.abs(torch.sin((phase1 - phase2) / 2))
            return 1 - torch.mean(distance, dim=-1)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def score_func(self, head, relation, tail, mode='single'):
        """
        Compute the score for the given triple (head, relation, tail).
        """
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE,
            'HAKE': self.HAKE,
        }
        
        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score

    def TransE(self, head, relation, tail, mode):
        """
        Compute the score using the TransE model.
        """
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        """
        Compute the score using the DistMult model.
        """
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        """
        Compute the score using the ComplEx model.
        """
        head_re, head_im = torch.chunk(head, 2, dim=2)
        relation_re, relation_im = torch.chunk(relation, 2, dim=2)
        tail_re, tail_im = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = relation_re * tail_re + relation_im * tail_im
            im_score = relation_re * tail_im - relation_im * tail_re
            score = head_re * re_score + head_im * im_score
        else:
            re_score = head_re * relation_re - head_im * relation_im
            im_score = head_re * relation_im + head_im * relation_re
            score = re_score * tail_re + im_score * tail_im

        score = score.sum(dim = 2)
        return score

    def RotatE(self, head, relation, tail, mode):
        """
        Compute the score using the RotatE model.
        """
        pi = 3.14159265358979323846
        
        head_re, head_im = torch.chunk(head, 2, dim=2)
        tail_re, tail_im = torch.chunk(tail, 2, dim=2)

        phase_relation = relation/(self.embedding_range.item()/pi)

        relation_re = torch.cos(phase_relation)
        relation_im = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = relation_re * tail_re + relation_im * tail_im
            im_score = relation_re * tail_im - relation_im * tail_re
            re_score = re_score - head_re
            im_score = im_score - head_im
        else:
            re_score = head_re * relation_re - head_im * relation_im
            im_score = head_re * relation_im + head_im * relation_re
            re_score = re_score - tail_re
            im_score = im_score - tail_im

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score

    def pRotatE(self, head, relation, tail, mode):
        """
        Compute the score using the pRotatE model.
        """
        pi = 3.14159262358979323846
        
        phase_head = head/(self.embedding_range.item()/pi)
        phase_relation = relation/(self.embedding_range.item()/pi)
        phase_tail = tail/(self.embedding_range.item()/pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)            
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim = 2) * self.modulus
        return score
    
    def HAKE(self, head, rel, tail, mode):
        """
        Compute the score using the HAKE model.
        """
        pi = 3.14159262358979323846
        
        phase_head, mod_head = torch.chunk(head, 2, dim=2)
        phase_relation, mod_relation, bias_relation = torch.chunk(rel, 3, dim=2)
        phase_tail, mod_tail = torch.chunk(tail, 2, dim=2)

        phase_head = phase_head / (self.embedding_range.item() / pi)
        phase_relation = phase_relation / (self.embedding_range.item() / pi)
        phase_tail = phase_tail / (self.embedding_range.item() / pi)

        if mode == 'head-batch':
            phase_score = phase_head + (phase_relation - phase_tail)
        else:
            phase_score = (phase_head + phase_relation) - phase_tail

        mod_relation = torch.abs(mod_relation)
        bias_relation = torch.clamp(bias_relation, max=1)
        indicator = (bias_relation < -mod_relation)
        bias_relation[indicator] = -mod_relation[indicator]

        r_score = mod_head * (mod_relation + bias_relation) - mod_tail * (1 - bias_relation)

        phase_score = torch.sum(torch.abs(torch.sin(phase_score / 2)), dim=2) * self.phase_weight
        r_score = torch.norm(r_score, dim=2) * self.modulus_weight

        return self.gamma.item() - (phase_score + r_score)
    
    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()
        
        ## Negative Samples
        if mode == 'head-batch':
            pass
        elif mode == 'tail-batch':
            pass

        text_dist_n, negative_score = model((positive_sample, negative_sample), mode=mode)

        if args.negative_adversarial_sampling:
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        ## Positive Sample
        positive_score = model(positive_sample, mode='single')
        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        ## Loss function
        loss = (positive_sample_loss + negative_sample_loss)/2
        
        entity_embedding = model.get_entity_embedding()
        
        if args.regularization != 0.0:
            regularization = args.regularization * (
                entity_embedding.norm(p = 3)**3 + 
                model.relation_embedding.norm(p = 3).norm(p = 3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
            
        loss = model.zeta_3 * loss + model.zeta_2 * (text_dist_n)
        
        loss = loss.mean()        
        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }
        
        loss_details = {
            'text_dist_n': text_dist_n.mean().item(),
        }
        
        log.update(loss_details)
        log_file_path = args.log_file_path  # 这里你需要确保args中有log_file_path属性  
        with open(log_file_path, 'a') as log_file:  # 以追加模式打开文件  
             log_file.write(str(log) + "\n")  # 将log字典转换为字符串并写入文件  
        return log
    
    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        model.eval()
    
        #Prepare dataloader for evaluation
        test_dataloader_head = DataLoader(
            TestDataset(
                test_triples, 
                all_true_triples, 
                args.nentity, 
                args.nrelation, 
                'head-batch',
                rerank=True if args.rerank == "true" else False
            ), 
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num//2), 
            collate_fn=TestDataset.collate_fn
        )

        test_dataloader_tail = DataLoader(
            TestDataset(
                test_triples, 
                all_true_triples, 
                args.nentity, 
                args.nrelation, 
                'tail-batch',
                rerank=True if args.rerank == "true" else False
            ), 
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num//2), 
            collate_fn=TestDataset.collate_fn
        )
        
        test_dataset_list = [test_dataloader_head, test_dataloader_tail]
        
        logs = []

        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, filter_bias, mode in tqdm(test_dataset):
                            
                    if args.cuda:
                        positive_sample = positive_sample.cuda()
                        negative_sample = negative_sample.cuda()
                        filter_bias = filter_bias.cuda()

                    batch_size = positive_sample.size(0)
                    
                    ## Negative Samples
                    if mode == 'head-batch':
                        pass
                    elif mode == 'tail-batch':
                        pass

                    text_dist, link_pred_score = model((positive_sample, negative_sample), mode)
                   
                    if args.fuse_score == "true":
                        if mode == 'head-batch':
                            tail = torch.index_select(model.get_entity_embedding(), dim=0, index=positive_sample[:, 2])
                            relation = torch.index_select(model.relation_embedding, dim=0, index=positive_sample[:, 1])
                            tail_relation = tail[:, None, :] - relation[:, None, :]
                            cosine_sim = F.cosine_similarity(tail_relation, model.get_entity_embedding()[None, :, :], dim=-1)
                        elif mode == 'tail-batch':
                            head = torch.index_select(model.get_entity_embedding(), dim=0, index=positive_sample[:, 0])
                            relation = torch.index_select(model.relation_embedding, dim=0, index=positive_sample[:, 1])
                            head_relation = head[:, None, :] + relation[:, None, :]
                            cosine_sim = F.cosine_similarity(head_relation, model.get_entity_embedding()[None, :, :], dim=-1)
                        else:
                            raise ValueError('mode %s not supported' % mode)
                        
                        link_pred_score += filter_bias
                        cosine_sim += filter_bias

                        normalized_score = F.normalize(link_pred_score, p=2, dim=-1)
                        normalized_cosine_sim = F.normalize(cosine_sim, p=2, dim=-1)

                        link_pred_rank = torch.argsort(normalized_score, dim=1, descending=True).argsort(dim=1)
                        cosine_sim_rank = torch.argsort(normalized_cosine_sim, dim=1, descending=True).argsort(dim=1)

                        link_pred_reciprocal_rank = 1.0 / (link_pred_rank.float() + 1)
                        cosine_sim_reciprocal_rank = 1.0 / (cosine_sim_rank.float() + 1)

                        combined_reciprocal_rank = 9.0 * link_pred_reciprocal_rank + 1.0 * cosine_sim_reciprocal_rank

                        argsort = torch.argsort(combined_reciprocal_rank, dim=1, descending=True)
                        
                    else:
                        score = link_pred_score + filter_bias
                        argsort = torch.argsort(score, dim=1, descending=True)

                    if mode == 'head-batch':
                        positive_arg = positive_sample[:, 0]
                    elif mode == 'tail-batch':
                        positive_arg = positive_sample[:, 2]
                    else:
                        raise ValueError('mode %s not supported' % mode)

                    for i in range(batch_size):
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        assert ranking.size(0) == 1

                        ranking = 1 + ranking.item()
                        logs.append({
                            'MRR': 1.0/ranking,
                            'MR': float(ranking),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@5': 1.0 if ranking <= 5 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                        })

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                    step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics
