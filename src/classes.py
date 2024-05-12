import math
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from scipy.special import softmax
import torch


class Metrics:
    def __init__(self, metrics):
        self.metrics = metrics
        self.f1 = False
        self.accuracy = False
        self._construct_metrics()

    def __call__(self, pred):
        return self._compute_metrics(pred)

    def _construct_metrics(self):
        if 'f1' in self.metrics:
            self.f1 = True
        if 'accuracy' in self.metrics:
            self.accuracy = True

    def _compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        metrics_data = {}

        if self.f1:
            f1 = f1_score(labels, preds, average="weighted")
            metrics_data['f1'] = f1
        if self.accuracy:
            acc = accuracy_score(labels, preds)
            metrics_data['accuracy'] = acc

        # logger.info(f"{metrics_data}")
        return metrics_data


class OnnxPipeline:
    def __init__(self, model, tokenizer, id2label, problem_type=None, return_all=True):
        self.model = model
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.return_all = return_all
        self.problem_type = problem_type

    def __call__(self, query):
        model_inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        inputs_onnx = {k: v.cpu().detach().numpy() for k, v in model_inputs.items()}
        logits = self.model.run(None, inputs_onnx)[0][0, :]

        if self.problem_type != 'multi_label_classification':
            probs = softmax(logits)
            if self.return_all == False:
                pred_idx = np.argmax(probs).item()
                return [{"label": self.id2label[pred_idx], "score": probs[pred_idx]}]
            else:
                return dict(zip(self.id2label.values(), probs))
        else:
            predicted_labels = self.__multilabel_preds([logits.tolist()])[0]
            return dict(zip(self.id2label.values(), predicted_labels))


    def __multilabel_preds(self, preds):
        """
        Transforms predictions output with sigmoid in to multilabel predictions

        :param predictions:     trainer predictions
        :return:                list of predictions for each label
        """

        result = []
        for pred in preds:
            prediction = []
            for x in list(pred):
                res = 1 / (1 + math.exp(-x))
                prediction.append(res)
            result.append(prediction)

        return result


class MultiLabelDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


