import torch
import numpy as np
from scipy.stats import norm
from tqdm import tqdm

class LiRAAttack:
    def __init__(self, shadow_models, n_queries=64):
        self.shadow_models = shadow_models
        self.n_queries = n_queries

    def fit(self, shadow_loader, is_member):
        self.member_scores = []
        self.non_member_scores = []
        
        for shadow_model in self.shadow_models:
            shadow_model.eval()
        
        with torch.no_grad():
            for batch in tqdm(shadow_loader, desc="Fitting LiRA"):
                graphs, labels = batch
                batch_scores = []
                
                for _ in range(self.n_queries):
                    logits = [model(graphs) for model in self.shadow_models]
                    scores = torch.stack([l[torch.arange(len(labels)), labels] for l in logits], dim=1)
                    batch_scores.append(scores)
                
                batch_scores = torch.stack(batch_scores, dim=1).mean(dim=1)
                
                for score, is_mem in zip(batch_scores, is_member):
                    if is_mem:
                        self.member_scores.append(score.cpu().numpy())
                    else:
                        self.non_member_scores.append(score.cpu().numpy())
        
        self.member_params = self._fit_gaussian(self.member_scores)
        self.non_member_params = self._fit_gaussian(self.non_member_scores)

    def predict(self, target_model, target_loader):
        target_model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in tqdm(target_loader, desc="LiRA Prediction"):
                graphs, labels = batch
                batch_scores = []
                
                for _ in range(self.n_queries):
                    logits = target_model(graphs)
                    scores = logits[torch.arange(len(labels)), labels]
                    batch_scores.append(scores)
                
                batch_scores = torch.stack(batch_scores, dim=1).mean(dim=1)
                
                for score in batch_scores:
                    member_likelihood = self._gaussian_likelihood(score.item(), *self.member_params)
                    non_member_likelihood = self._gaussian_likelihood(score.item(), *self.non_member_params)
                    likelihood_ratio = member_likelihood / non_member_likelihood
                    predictions.append(likelihood_ratio > 1)
        
        return np.array(predictions)

    def _fit_gaussian(self, data):
        return np.mean(data), np.std(data)

    def _gaussian_likelihood(self, x, mean, std):
        return norm.pdf(x, mean, std)