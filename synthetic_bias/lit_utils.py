r"""Code example for a custom model, using PyTorch.
This demo shows how to use a custom model with LIT, in just a few lines of code.
We'll use a transformers model, with a minimal amount of code to implement the
LIT API. Compared to models/glue_models.py, this has fewer features, but the
code is more readable.
This demo is equivalent in functionality to simple_tf2_demo.py, but uses PyTorch
instead of TensorFlow 2. The models behave identically as far as LIT is
concerned, and the implementation is quite similar - to see changes, run:
  git diff --no-index simple_tf2_demo.py simple_pytorch_demo.py
The transformers library can load weights from either,
so you can use any saved model compatible with the underlying model class
(AutoModelForSequenceClassification). To train something for this demo, you can:
- Use quickstart_sst_demo.py, and set --model_path to somewhere durable
- Or: Use tools/glue_trainer.py
- Or: Use any fine-tuning code that works with transformers, such as
https://github.com/huggingface/transformers#quick-tour-of-the-fine-tuningusage-scripts
To run locally:
  python -m lit_nlp.examples.simple_pytorch_demo \
      --port=5432 --model_path=/path/to/saved/model
Then navigate to localhost:5432 to access the demo UI.
NOTE: this demo still uses TensorFlow Datasets (which depends on TensorFlow) to
load the data. However, the output of glue.SST2Data is just NumPy arrays and
plain Python data, and you can easily replace this with a different library or
directly loading from CSV.
"""

from lit_nlp import dev_server
from lit_nlp.api import model as lit_model
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types
from lit_nlp.lib import utils

import torch
from torch.autograd import grad
import numpy as np


class NLIDatasetWrapper(lit_dataset.Dataset):
    """Loader for MultiNLI development set."""

    NLI_LABELS = ['entailment', 'neutral', 'contradiction']
    NLI_LABELS.sort()

    def __init__(self, ds, bias_type, bias_token):
        # Store as a list of dicts, conforming to self.spec()
        examples = []
        for ind in range(len(ds)):
            p, h, y = ds[ind]
            examples.append({'premise': p, 'hypothesis': h, 'label': self.NLI_LABELS[y],
                             'bias_type': bias_type, 'bias_token': bias_token, 'grad_class': self.NLI_LABELS[y]})
        self._examples = examples

    def spec(self):
        return {
            'premise': lit_types.TextSegment(),
            'hypothesis': lit_types.TextSegment(),
            'label': lit_types.CategoryLabel(vocab=self.NLI_LABELS),
            'bias_type': lit_types.CategoryLabel(vocab=self.NLI_LABELS + ['none']),
            'bias_token': lit_types.TextSegment(), #lit_types.CategoryLabel(),
            'grad_class': lit_types.CategoryLabel(vocab=self.NLI_LABELS),
        }


class NLIModelWrapper(lit_model.Model):
    LABELS = ['entailment', 'neutral', 'contradiction']
    LABELS.sort()

    def __init__(self, tokenizer, model, bs=32):
        self.batch_size = bs
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()

    # LIT API implementation
    def max_minibatch_size(self):
        # This tells lit_model.Model.predict() how to batch inputs to
        # predict_minibatch().
        # Alternately, you can just override predict() and handle batching yourself.
        return self.batch_size

    def predict_minibatch(self, inputs):
        p, h = list(map(lambda x: x['premise'], inputs)), list(map(lambda x: x['hypothesis'], inputs))
        input_dict = self.tokenizer(p, h, padding=True, truncation=True, return_tensors='pt')  # tensors - shape B x S
        batched_outputs = {}

        # Check and send to cuda (GPU) if available
        if torch.cuda.is_available():
            self.model.cuda()
            for tensor in input_dict:
                input_dict[tensor] = input_dict[tensor].cuda()

        # for integrated gradients - precalculate word embeddings and pass them in instead of input_ids to enable
        # .grad of output with respect to embeddings
        input_ids = input_dict["input_ids"]
        word_embeddings = self.model.bert.embeddings.word_embeddings
        input_embs = word_embeddings(input_ids)  # tensor of shape B x S x h
        input_embs = scatter_embs(input_embs, inputs)
        model_inputs = input_dict.copy()
        model_inputs["input_ids"] = None

        logits, hidden_states, unused_attentions = self.model(**model_inputs, inputs_embeds=input_embs)

        # for integrated gradients - Choose output to "explain"(from num_labels) according to grad_class
        grad_classes = [self.LABELS.index(ex["grad_class"]) for ex in inputs]  # list of length B of integer indices
        indices = np.arange(len(grad_classes)).tolist(), grad_classes
        scalar_pred_for_gradients = logits[indices]

        # prepare model outputs according to output_spec
        # all values must be numpy arrays or tensor (have the shape attribute) in order to unbatch them
        batched_outputs["input_emb_grads"] = grad(scalar_pred_for_gradients, input_embs,
                                                  torch.ones_like(scalar_pred_for_gradients))[0].detach().to('cpu').numpy()

        batched_outputs["probas"] = torch.nn.functional.softmax(logits, dim=-1).detach().to('cpu').numpy()  # B x num_labels
        batched_outputs["input_ids"] = input_dict["input_ids"].detach().to('cpu').numpy()  # B x S
        batched_outputs["cls_pooled"] = hidden_states[-1][:, 0, :].detach().to('cpu').numpy()  # output of embeddings layer, B x h
        batched_outputs["input_embs"] = input_embs.detach().to('cpu').numpy()  # B x S x h
        batched_outputs["grad_class"] = np.array([ex["grad_class"] for ex in inputs])

        # Unbatch outputs so we get one record per input example.
        for output in utils.unbatch_preds(batched_outputs):
            output["tokens"] = self.tokenizer.convert_ids_to_tokens(output.pop("input_ids"))  # list of length seq
            output = self._postprocess(output)
            yield output

    def _postprocess(self, output_samp):
        special_tokens_mask = list(map(lambda x: x != self.tokenizer.pad_token, output_samp['tokens']))
        output_samp['tokens'] = (np.array(output_samp['tokens'])[special_tokens_mask]).tolist()
        output_samp['input_embs'] = output_samp['input_embs'][special_tokens_mask]
        output_samp['input_emb_grads'] = output_samp['input_emb_grads'][special_tokens_mask]
        return output_samp

    def input_spec(self) -> lit_types.Spec:
        inputs = {}
        inputs["premise"] = lit_types.TextSegment()
        inputs["hypothesis"] = lit_types.TextSegment()

        # for gradient attribution
        inputs["input_embs"] = lit_types.TokenEmbeddings(required=False)
        inputs["grad_class"] = lit_types.CategoryLabel(vocab=self.LABELS)

        return inputs

    def output_spec(self) -> lit_types.Spec:
        output = {}
        output["tokens"] = lit_types.Tokens()
        output["probas"] = lit_types.MulticlassPreds(parent="label", vocab=self.LABELS)
        output["cls_pooled"] = lit_types.Embeddings()

        # for gradient attribution
        output["input_embs"] = lit_types.TokenEmbeddings()
        output["grad_class"] = lit_types.CategoryLabel(vocab=self.LABELS)
        output["input_emb_grads"] = lit_types.TokenGradients(align="tokens",
                                                             grad_for="input_embs", grad_target="grad_class")

        return output


def scatter_embs(input_embs, inputs):
    """
    For inputs that have 'input_embs' field passed in, replace the entry in input_embs[i] with the entry
    from inputs[i]['input_embs']. This is useful for the Integrated Gradients - for which the predict is
    called with inputs with 'input_embs' field which is an interpolation between the baseline and the real calculated
    input embeddings for the sample.
    :param input_embs: tensor of shape B x S x h of input embeddings according to the input sentences.
    :param inputs: list of dictionaries (smaples), for which the 'input_embs' field might be specified
    :return: tensor of shape B x S x h with embeddings (if passed) from inputs inserted to input_embs
    """
    interp_embeds = [(ind, ex.get('input_embs')) for ind, ex in enumerate(inputs)]
    for ind, embed in interp_embeds:
        if embed is not None:
            input_embs[ind] = torch.tensor(embed)

    return input_embs
