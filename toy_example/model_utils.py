# Simple model architecture, as described in "Mitigating bias in NLI". Randomly initialized embedding layers for
# premise and hypothesis. bag of words representation (summing over embeddings of the words, i.e over sequence
# length). Then 1 hidden layer (option for more) followed by ReLU, then linear classifier.
import torch
import torch.nn as nn
import torch.optim


class NLIFeatureExtractor(nn.Module):
    def __init__(self, embed_hypothesis, num_layers=1, hidden_dim=256):
        super().__init__()
        self.embed_premise = nn.Embedding(embed_hypothesis.num_embeddings, embed_hypothesis.embedding_dim)
        self.embed_hypothesis = embed_hypothesis
        self.features = self.init_features(num_layers, hidden_dim)

    def init_features(self, num_layers, hidden_dim):
        layers, hidden_dim = [], num_layers * [hidden_dim]
        input_dim = self.embed_hypothesis.embedding_dim + self.embed_premise.embedding_dim
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim[i]))
            layers.append(nn.ReLU())
            input_dim = hidden_dim[i]

        return nn.Sequential(*layers)

    def forward(self, batch):
        """batch : tuple where each element is of dims B x S
        embedded_p, embedded_h after embedding: B x S x E , sum over sentence dim -> B x 1 x E ,
        concatenate embedded hypothesis and premise before input to classifier:  B x (2 x E)."""
        batch_p, batch_h = batch
        batch_dim = batch_p.shape[0]
        embedded_p = self.embed_premise(batch_p).sum(dim=1, keepdim=True).view(batch_dim, -1)
        embedded_h = self.embed_hypothesis(batch_h).sum(dim=1, keepdim=True).view(batch_dim, -1)
        output = self.features(torch.cat([embedded_p, embedded_h], 1))
        return output


class NLINet(NLIFeatureExtractor):
    def __init__(self, embed_hypothesis, num_layers=1, hidden_dim=256, multi_class=False):
        super().__init__(embed_hypothesis, num_layers=num_layers, hidden_dim=hidden_dim)
        if multi_class:
            self.output_dim = 2
        else:
            self.output_dim = 1
        self.classifier = self.init_classifier(hidden_dim)

    def init_classifier(self, hidden_dim):
        layers = []
        input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, self.output_dim))

        return nn.Sequential(*layers)

    def forward(self, batch):
        """batch : tuple where each element is of dims B x S
        embedded_p, embedded_h after embedding: B x S x E , sum over sentence dim -> B x 1 x E ,
        concatenate embedded hypothesis and premise before input to classifier:  B x (2 x E)."""
        features = super().forward(batch)
        output = self.classifier(features)
        return output, features