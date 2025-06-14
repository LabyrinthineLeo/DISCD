# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json, os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)


class DISCD(nn.Module):
    def __init__(self, student_n, exer_n, knowledge_n, inter_matrix, qmatrix_path):
        self.model_name = "discd"
        self.stu_num = student_n
        self.que_num = exer_n
        self.cpt_num = knowledge_n

        self.hidden_size = 128
        self.prednet_input_len = knowledge_n
        self.prednet_len1, self.prednet_len2 = 256, 128  # changeable

        super(DISCD, self).__init__()

        self.stu_emb_phi = nn.Embedding(self.stu_num, self.cpt_num)
        self.stu_emb_theta = nn.Embedding(self.stu_num, self.cpt_num)
        self.stu_emb_omega = nn.Embedding(self.stu_num, self.cpt_num)

        self.k_difficulty = nn.Embedding(self.que_num, self.cpt_num)
        self.e_discrimination = nn.Embedding(self.que_num, 1)
        self.alpha = nn.Parameter(torch.tensor(1.))

        # encoders
        self.phi_layer = nn.Sequential(
            nn.Linear(self.cpt_num, 128),
            nn.Sigmoid(),
            nn.Linear(128, 2*self.hidden_size),
        )

        self.phi_layer_output = nn.Linear(self.hidden_size, self.cpt_num)
        self.theta_layer = nn.Linear(self.cpt_num+1, 2*self.cpt_num*self.hidden_size)

        self.theta_layer_output = nn.Linear(self.cpt_num, self.cpt_num)
        self.omega_list = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(64, 32, 3, padding=1),
            nn.LeakyReLU()
        )
        self.omega_layer = nn.Linear(32, 2*self.cpt_num)

        self.cross_attn = CrossAttention(self.hidden_size, self.hidden_size)

        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)

        # interaction matrix
        self.interact_matrix = inter_matrix.to(device)

        # Q-Matrix
        q_matrix_json = json.load(open(qmatrix_path, 'r'))
        q_matrix = torch.zeros((self.que_num, self.cpt_num), dtype=torch.float32).to(device)
        for exer_id, cpt_list in q_matrix_json.items():
            for cpt_id in cpt_list:
                q_matrix[int(exer_id), cpt_id] = 1.
        self.q_matrix = q_matrix

        # knowledge matrix
        interact_matrix_sparse = self.interact_matrix.to_sparse()
        knowledge_matrix = torch.sparse.mm(interact_matrix_sparse, self.q_matrix)
        km_stand = (knowledge_matrix - knowledge_matrix.min()) / (knowledge_matrix.max()-knowledge_matrix.min())
        self.knowledge_matrix = km_stand

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def encoder_phi(self, phi_input):
        x = self.phi_layer(phi_input)
        mu_phi, log_var_phi = torch.chunk(x, 2, dim=1)
        return mu_phi, log_var_phi

    def encoder_theta(self, theta_input):
        x = F.sigmoid(self.theta_layer(theta_input)).reshape((-1, self.cpt_num, 2*self.hidden_size))
        mu_theta, log_var_theta = torch.chunk(x, 2, dim=2)
        return mu_theta, log_var_theta

    def encoder_omega(self, omega_input):

        mask = (omega_input != -1).float()
        omega_x = omega_input.clone()
        omega_x[omega_x == -1] = 0

        omega_x = omega_x.unsqueeze(1)
        x = self.omega_list(omega_x)
        x = x * mask.unsqueeze(1)
        x = F.adaptive_avg_pool1d(x, 1)

        x = self.omega_layer(x.view(x.shape[0], -1))
        mu_omega, log_var_omega = torch.chunk(x, 2, dim=1)
        return mu_omega, log_var_omega

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def kl_loss(self, mu_phi, log_var_phi, mu_theta, log_var_theta, mu_omega, log_var_omega):
        kl_phi = -0.5 * torch.mean(1 + log_var_phi - mu_phi.pow(2) - log_var_phi.exp())
        kl_theta = -0.5 * torch.mean(1 + log_var_theta - mu_theta.pow(2) - log_var_theta.exp())
        kl_omega = -0.5 * torch.mean(1 + log_var_omega - mu_omega.pow(2) - log_var_omega.exp())

        total_loss = kl_phi + kl_theta + kl_omega
        return total_loss

    def forward(self, stu_id, que_id, cpt_list, training=True):
        # phi input
        phi_input = self.stu_emb_phi(stu_id)
        mu_phi, log_var_phi = self.encoder_phi(phi_input)

        # theta input
        theta_input = torch.cat([self.stu_emb_theta(stu_id),torch.mean(self.knowledge_matrix[stu_id],dim=-1,keepdim=True)],dim=-1)  # [bsz, K+1] 拼接

        mu_theta, log_var_theta = self.encoder_theta(theta_input)

        # omega input
        omega_input = self.interact_matrix[stu_id]
        mu_omega, log_var_omega = self.encoder_omega(omega_input)

        if training:
            z_phi = self.reparameterize(mu_phi, log_var_phi)
            z_theta = self.reparameterize(mu_theta, log_var_theta)
            z_omega = self.reparameterize(mu_omega, log_var_omega)
        else:
            z_phi = mu_phi
            z_theta = mu_theta
            z_omega = mu_omega

        kl_losses = self.kl_loss(mu_phi, log_var_phi, mu_theta, log_var_theta, mu_omega, log_var_omega)

        z_hat = self.cross_attn(torch.sigmoid(z_phi), torch.sigmoid(z_theta))
        z = z_hat + 1.*torch.tanh(z_omega)

        stat_emb = torch.sigmoid(z)
        k_difficulty = torch.sigmoid(self.k_difficulty(que_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(que_id))

        input_x = e_discrimination * (stat_emb - k_difficulty) * cpt_list
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output = torch.sigmoid(self.prednet_full3(input_x)).reshape(-1)

        return output, kl_losses, 0.


class CrossAttention(nn.Module):
    def __init__(self, d_q, d_k):
        super(CrossAttention, self).__init__()
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_k

        # projection
        self.query_proj = nn.Linear(d_q, d_k)
        self.key_proj = nn.Linear(d_k, d_k)

    def forward(self, query, key):
        query = query.unsqueeze(1)
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = key

        scores = torch.bmm(query, key.transpose(1, 2)) / (self.d_k ** 0.5)
        att_weights = torch.softmax(scores, dim=-1)
        weight_value = att_weights.transpose(1, 2) * value
        output = torch.mean(weight_value, dim=-1)

        return output
