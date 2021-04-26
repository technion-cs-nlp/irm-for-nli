import torch
from copy import deepcopy
from sklearn.metrics import confusion_matrix
from captum.attr import visualization as viz
from captum.attr import IntegratedGradients, LayerIntegratedGradients
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer

import settings
from utils import attribution_to_heatmap_figure, wrap_confusion_matrix
import numpy as np
from collections import Counter


def construct_input_ref_pair(tokenizer, premise, hypothesis, device='cpu'):
    """Construct inputs (input_ids, tokentype_ids and position_ids)) to input to model for integrated gradients.
     Computes the real inputs (according to the passed premise and hypothesis) and constructs the reference inputs
     accordingly.
     Reference inputs are as follows: all zeros for tokentype_ids. all zeros for position_ids (to not indicate
      position or belongign to some sequence), for input_ids replace all tokens that are not [CLS] or [SEP] with
      [PAD] tokens."""
    ref_token_id, sep_token_id, cls_token_id = tokenizer.vocab[tokenizer.pad_token], \
                                               tokenizer.vocab[tokenizer.sep_token], \
                                               tokenizer.vocab[tokenizer.cls_token]

    input_dict = tokenizer(premise, hypothesis, padding=True, truncation=True, return_tensors='pt')
    input_ids, token_type_ids, attention_mask = input_dict['input_ids'], input_dict['token_type_ids'], input_dict[
        'attention_mask']
    batch_size, seq_len = input_ids.shape
    position_ids = torch.tensor([[i for i in range(seq_len)] for k in range(batch_size)])

    # construct reference inputs
    premise_lengths = [seq.index(sep_token_id) - 1 for seq in input_ids.tolist()]
    hypothesis_lengths = (
            seq_len - 3 - torch.tensor(premise_lengths)).tolist()  # 3 special tokens - [cls] ... [sep] ... [sep]
    ref_input_ids = deepcopy(input_ids)
    ref_input_ids[torch.bitwise_and(ref_input_ids != sep_token_id, ref_input_ids != cls_token_id)] = ref_token_id

    ref_token_type_ids = torch.zeros_like(token_type_ids)
    # we could potentially also use random permutation with `torch.randperm(seq_length, device=device)`
    ref_position_ids = torch.zeros_like(position_ids)

    return (input_ids.to(device=device), token_type_ids.to(device=device),
            position_ids.to(device=device)), \
           (ref_input_ids.to(device=device), ref_token_type_ids.to(device=device),
            ref_position_ids.to(device=device)), attention_mask.to(device=device)


def get_bias_indices(input_ids, tokenizer):
    """Return a list of bias indices (assuming bias was added to hypothesis) :
    the index is the one after the first [SEP] token"""
    input_ids_list = input_ids.to('cpu').tolist()
    sep_token_id = tokenizer.sep_token_id
    tokens_len = [len(tokenizer(x)['input_ids']) - 2 for x in settings.VOCAB_BIAS[0]]
    assert len(set(tokens_len)) == 1, 'Different bias tokens are tokenizer to different number of subtokens'
    bias_ind = []
    for seq in input_ids_list:
        first_sep_ind = seq.index(sep_token_id)
        bias_ind.append([first_sep_ind + x for x in range(1, tokens_len[0])])

    return bias_ind


def summarize_attributions(attributions):
    attributions = attributions.sum(dim=2)  # B x S x h ... => B x S ...
    attributions = attributions / torch.norm(attributions, p=2, dim=1).unsqueeze(1)  # B x S ...
    return attributions


def attributions_cm(attr, y_gt, y, bias_indices, num_labels):
    """
    Generate a "confusion matrix" from embedding attribution. The attributions with respect to the bias tokens are used
    as weights for the "confusion matrix"
    """
    B, S = attr.shape  # attr shape: B x S
    bias_attr = attr[torch.arange(B).unsqueeze(-1), bias_indices]  # B x 3 ( bias token is tokenized to 3 : [<, char, >])
    bias_attr = torch.sum(bias_attr, dim=-1)  # B
    with wrap_confusion_matrix(num_labels, y_gt, y, bias_attr) as padded_inputs:
        cm = confusion_matrix(y_true=padded_inputs[0], y_pred=padded_inputs[1], sample_weight=padded_inputs[2])
    return cm


def confusion_matrix_from_embedding_attribution(model, tokenizer, dl, device='cpu', internal_bs=None):
    """
    Construct "confusion matrix" from bias token embedding attributions.
    For each sample, attribute gold label and predicted label with respect to input embeddings.
    Convert the embeddings attributions to score per token by summing across embeddings dimension
    and normalize along the sequence dimension.
    Retain only bias token embedding attribution score.
    :param model: The model for which the predictions are explained
    :param tokenizer: tokenizer to use with the model
    :param dl: Dataloader built on top of the Dataset to attribute
    :param device: cpu/gpu
    :param internal_bs: Additional parameter passed to captum attribution method. Enables splitting the attribution
    to smaller batches (not evaluating all steps at once but only some of the steps every time). Default: None
    :return:  attribution_cm_dict['predicted label'], attribution_cm_dict['gold label'] - each confusion matrix is a
    numpy array
    attribution_dict['predicted label'], attribution_dict['gold label'] - 2 list specifying the bias token attribution per sample
    """
    model.eval()
    num_labels = model.config.num_labels

    def predict(input_ids, token_type_ids, position_ids, attention_mask_ids=None):
        logits = model(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids,
                       attention_mask=attention_mask_ids)[0]
        return logits

    ig = IntegratedGradients(predict)
    interpretable_word_embedding = configure_interpretable_embedding_layer(model, 'bert.embeddings.word_embeddings')
    interpretable_tokentype_embedding = configure_interpretable_embedding_layer(model,
                                                                                'bert.embeddings.token_type_embeddings')
    interpretable_position_embedding = configure_interpretable_embedding_layer(model,
                                                                               'bert.embeddings.position_embeddings')

    attribution_cm_dict, cm = {'gold label': None, 'predicted label': None}, None
    attribution_dict = {'gold label': None, 'predicted label': None}
    for batch in dl:
        # batch =  [p, h, y]. p, h : 2 tuples each of length B (batch size), y : tensor of shape B

        # construct original input and refernece inputs
        inputs, ref_inputs, attention_mask = construct_input_ref_pair(tokenizer, batch[0], batch[1], device)
        bias_ind = get_bias_indices(inputs[0], tokenizer)  # list of len B x 3
        # input_ids, tokentype_ids, position_ids = inputs
        # ref_input_ids, ref_tokentype_ids, ref_position_ids = ref_inputs

        # generate input and baseline embeddings - for baseline keep embedding of original tokentype and position,
        # zero out word embedding
        word_embeddings = interpretable_word_embedding.indices_to_embeddings(inputs[0])
        ref_word_embeddings = interpretable_word_embedding.indices_to_embeddings(ref_inputs[0])
        # ref_word_embeddings = torch.zeros_like(word_embeddings).to(device=device)
        tokentype_embeddings = interpretable_tokentype_embedding.indices_to_embeddings(inputs[1])
        position_embeddings = interpretable_position_embedding.indices_to_embeddings(inputs[2])

        # attributions per label
        with torch.no_grad():
            logits = predict(word_embeddings, tokentype_embeddings, position_embeddings, attention_mask)
        softmax = torch.nn.Softmax(dim=-1)
        preds = torch.argmax(softmax(logits),
                             dim=-1)  # tensor of length B - these are the targets (models predictions to explain)
        with wrap_confusion_matrix(num_labels, batch[-1], preds.to('cpu')) as padded_inputs:
            y_gt, y, _ = padded_inputs
            if cm is None:
                cm = confusion_matrix(y_true=y_gt, y_pred=y)
            else:
                cm += confusion_matrix(y_true=y_gt, y_pred=y)
        for k, targets in zip(['predicted label', 'gold label'], [preds, batch[-1]]):
            attributions = ig.attribute(inputs=word_embeddings, baselines=ref_word_embeddings,
                                        additional_forward_args=(
                                            tokentype_embeddings, position_embeddings, attention_mask),
                                        target=targets.to(device), method="riemann_trapezoid",
                                        internal_batch_size=32, n_steps=101).detach().to('cpu')
            # tensor - B x S x h
            # zero gradient attribution for padded token embeddings
            expanded_attention_mask = attention_mask.to(device='cpu').unsqueeze(-1)
            attributions = attributions * expanded_attention_mask
            # covert embedding attribution to score : B x S x h  => B x S  (normalize across sequence dimension)
            attributions = summarize_attributions(attributions).to('cpu')
            preds = preds.to('cpu')

            if attribution_cm_dict[k] is None:
                attribution_cm_dict[k] = attributions_cm(attributions, batch[-1], preds, bias_ind, num_labels)
            else:
                attribution_cm_dict[k] += attributions_cm(attributions, batch[-1], preds, bias_ind, num_labels)
            bias_attr = attributions[torch.arange(len(bias_ind)).unsqueeze(-1), bias_ind]
            bias_attr = torch.sum(bias_attr, dim=-1).tolist() # list of len B
            if attribution_dict[k] is None:
                attribution_dict[k] = bias_attr
            else:
                attribution_dict[k] += bias_attr

    # normalize each row in the confusion matrix by number of samples : np array of shape num_labels x 1
    for k in ['predicted label', 'gold label']:
        attribution_cm_dict[k] = np.around(attribution_cm_dict[k] / (cm + np.ones_like(cm) * 1e-5), 2)

    remove_interpretable_embedding_layer(model, interpretable_word_embedding)
    remove_interpretable_embedding_layer(model, interpretable_tokentype_embedding)
    remove_interpretable_embedding_layer(model, interpretable_position_embedding)

    return attribution_cm_dict['predicted label'], attribution_cm_dict['gold label'], \
           attribution_dict['predicted label'], attribution_dict['gold label']


def heat_map_from_embedding_attribution1(model, tokenizer, samp, device='cpu'):
    """
    Qualitative analysis of gradient attribution with respect to input embeddings.
    For one sample - calculate gradients of each of the predictions with respect to the input embedding.
    Generate a heat map of the gradient attributions.
    In this implementation we don't have control over the embeddings - we can only control the tokens (so, for example,
    we can't force zero embedding).
    """
    model.eval()

    def predict(input_ids, token_type_ids=None, position_ids=None, attention_mask_ids=None):
        logits = model(input_ids, token_type_ids=token_type_ids,
                       position_ids=position_ids, attention_mask=attention_mask_ids)[0]
        return logits

    ig = LayerIntegratedGradients(predict, model.bert.embeddings)

    p, h, y = samp  # p, h : strings. y: tensor, shape []
    labels = ['contradiction', 'entailment', 'neutral']
    labels.sort()

    inputs, ref_inputs, attention_mask = construct_input_ref_pair(tokenizer, p, h, device)
    input_ids, tokentype_ids, position_ids = inputs
    ref_input_ids, ref_tokentype_ids, ref_position_ids = ref_inputs
    # inputs = (input_ids, tokentype_ids, position_ids) = (tensor: 1 x S, tensor: 1 x S, tensor: 1 x S)
    # same for ref_inputs
    # attention_mask : tensor : 1 x S

    attr_dict = dict()
    # attributions per label
    for lbl, label in enumerate(labels):
        label_attributions = ig.attribute(inputs=(input_ids, tokentype_ids, position_ids),
                                          baselines=(ref_input_ids, ref_tokentype_ids, ref_position_ids),
                                          additional_forward_args=(attention_mask,),
                                          target=torch.ones_like(y, device=device) * lbl, n_steps=30).detach().to('cpu')
        # tensor: 1 x S x h
        label_attributions = summarize_attributions(label_attributions)  # 1 x S x h => 1 x S
        attr_dict[label] = label_attributions

    fig = attribution_to_heatmap_figure(attr_dict, tokenizer.convert_ids_to_tokens(inputs[0].squeeze()))

    return fig


def heat_map_from_embedding_attribution2(model, tokenizer, samp, device='cpu'):
    """
    Qualitative analysis of gradient attribution with respect to input embeddings.
    For one sample - calculate gradients of each of the predictions with respect to the input embedding.
    Generate a heat map of the gradient attributions.
    In this version of implementation we can specifically control embeddings. So zero embedding can be forced
    (for example to compare to lit - in which the baseline is zeroing out only the embedding of the words and keeping
    the other two embeddings - token type and position)
    """
    model.eval()

    def predict(input_ids, token_type_ids, position_ids, attention_mask_ids=None):
        logits = model(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids,
                       attention_mask=attention_mask_ids)[0]
        return logits

    ig = IntegratedGradients(predict)

    p, h, y = samp  # p, h : strings. y: tensor, shape []
    labels = ['contradiction', 'entailment', 'neutral']
    labels.sort()

    inputs, ref_inputs, attention_mask = construct_input_ref_pair(tokenizer, p, h, device)
    # inputs = (input_ids, tokentype_ids, position_ids) = (tensor: 1 x S, tensor: 1 x S, tensor: 1 x S)
    # same for ref_inputs
    # attention_mask : tensor : 1 x S

    # generate input and baseline embeddings - for baseline keep embedding of original tokentype and position,
    # zero out word embedding
    interpretable_word_embedding = configure_interpretable_embedding_layer(model, 'bert.embeddings.word_embeddings')
    interpretable_tokentype_embedding = configure_interpretable_embedding_layer(model,
                                                                                'bert.embeddings.token_type_embeddings')
    interpretable_position_embedding = configure_interpretable_embedding_layer(model,
                                                                               'bert.embeddings.position_embeddings')
    word_embeddings = interpretable_word_embedding.indices_to_embeddings(inputs[0])
    ref_word_embeddings = torch.zeros_like(word_embeddings)
    tokentype_embeddings = interpretable_tokentype_embedding.indices_to_embeddings(inputs[1])
    position_embeddings = interpretable_position_embedding.indices_to_embeddings(inputs[2])

    attr_dict = dict()
    # attributions per label
    for lbl, label in enumerate(labels):
        # wrapped interpretable layer just passes input to output so pass the embeddings you want to attribute to
        label_attributions = ig.attribute(inputs=word_embeddings, baselines=ref_word_embeddings,
                                          additional_forward_args=(
                                              tokentype_embeddings, position_embeddings, attention_mask),
                                          target=torch.ones_like(y, device=device) * lbl, n_steps=31,
                                          method="riemann_trapezoid").detach().to('cpu')
        # tuple of sub layer attributions : tensor: 1 x S x h
        label_attributions = summarize_attributions(label_attributions)  # 1 x S x h => 1 x S
        attr_dict[label] = label_attributions

    remove_interpretable_embedding_layer(model, interpretable_word_embedding)
    remove_interpretable_embedding_layer(model, interpretable_tokentype_embedding)
    remove_interpretable_embedding_layer(model, interpretable_position_embedding)
    fig = attribution_to_heatmap_figure(attr_dict, tokenizer.convert_ids_to_tokens(inputs[0].squeeze()))

    return fig


def heat_map_from_sub_embedding_attribution(model, tokenizer, samp, device='cpu'):
    model.eval()

    def predict(input_ids, token_type_ids=None, position_ids=None, attention_mask_ids=None):
        logits = model(input_ids, token_type_ids=token_type_ids,
                       position_ids=position_ids, attention_mask=attention_mask_ids)[0]
        return logits

    ig = IntegratedGradients(predict)

    # wrap with interpretable layer
    interpretable_word_embedding = configure_interpretable_embedding_layer(model, 'bert.embeddings.word_embeddings')
    interpretable_tokentype_embedding = configure_interpretable_embedding_layer(model,
                                                                                'bert.embeddings.token_type_embeddings')
    interpretable_position_embedding = configure_interpretable_embedding_layer(model,
                                                                               'bert.embeddings.position_embeddings')

    p, h, y = samp  # p, h : strings. y: tensor, shape []
    labels = ['contradiction', 'entailment', 'neutral']
    labels.sort()

    inputs, ref_inputs, attention_mask = construct_input_ref_pair(tokenizer, p, h, device)
    # inputs = (input_ids, tokentype_ids, position_ids) = (tensor: 1 x S, tensor: 1 x S, tensor: 1 x S)
    # same for ref_inputs
    # attention_mask : tensor : 1 x S

    input_embeddings = interpretable_word_embedding.indices_to_embeddings(inputs[0]), \
                       interpretable_tokentype_embedding.indices_to_embeddings(inputs[1]), \
                       interpretable_position_embedding.indices_to_embeddings(inputs[2])
    ref_input_embeddings = interpretable_word_embedding.indices_to_embeddings(ref_inputs[0]), \
                           interpretable_tokentype_embedding.indices_to_embeddings(ref_inputs[1]), \
                           interpretable_position_embedding.indices_to_embeddings(ref_inputs[2])

    visualization_dict = dict()
    # attributions per label
    for lbl, label in enumerate(labels):
        label_attributions = ig.attribute(inputs=input_embeddings, baselines=ref_input_embeddings,
                                          additional_forward_args=(inputs[1], inputs[2], attention_mask),
                                          target=torch.ones_like(y, device=device) * lbl)
        # tuple of sub layer attributions : tensor: 1 x S x h
        label_attributions = [summarize_attributions(attr).to('cpu') for attr in
                              label_attributions]  # 1 x S x h => 1 x S

        sub_layer_vis_dict = dict()
        for i, k in enumerate(['word', 'tokentype', 'position']):
            sub_layer_vis_dict[k] = viz.VisualizationDataRecord(
                label_attributions[i],
                torch.tensor(1.0),
                torch.tensor(1),
                torch.tensor(1),
                torch.tensor(1),
                label_attributions[i].sum(),
                tokenizer.convert_ids_to_tokens(inputs[0]),
            )
            viz.visualize_text([sub_layer_vis_dict[k]])
        visualization_dict[label] = sub_layer_vis_dict

    remove_interpretable_embedding_layer(model, interpretable_word_embedding)
    remove_interpretable_embedding_layer(model, interpretable_tokentype_embedding)
    remove_interpretable_embedding_layer(model, interpretable_position_embedding)

    return visualization_dict
