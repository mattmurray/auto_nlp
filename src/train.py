from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial
import shutil

import settings.general
import src.utils as utils
from src.settings import general

from datasets import load_metric, load_dataset
from datasets import ClassLabel, Value

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers import Trainer, TrainingArguments
from transformers.convert_graph_to_onnx import convert

from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

from scipy.special import softmax
from onnxruntime import (GraphOptimizationLevel, InferenceSession, SessionOptions)
from onnxruntime.quantization import quantize_dynamic, QuantType
import os
from psutil import cpu_count

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

        logger.info(f"{metrics_data}")
        return metrics_data


class OnnxPipeline:
    def __init__(self, model, tokenizer, id2label, return_all=True):
        self.model = model
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.return_all = return_all

    def __call__(self, query):
        model_inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        inputs_onnx = {k: v.cpu().detach().numpy() for k, v in model_inputs.items()}
        logits = self.model.run(None, inputs_onnx)[0][0, :]
        probs = softmax(logits)
        if self.return_all == False:
            pred_idx = np.argmax(probs).item()
            return [{"label": self.id2label[pred_idx], "score": probs[pred_idx]}]
        else:
            return dict(zip(id2label.values(), probs))

def plot_confusion_matrix(y_preds, y_true, labels, normalized="true"):
    cm = confusion_matrix(y_true, y_preds, normalize=normalized)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    if normalized == 'true':
        disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    else:
        disp.plot(cmap="Blues", values_format=".0f", ax=ax, colorbar=False)

    plt.title("Test data confusion matrix")
    return plt


def create_model_for_provider(model_path, provider="CPUExecutionProvider"):
    # creates InterenceSession to feed inputs to the model
    options = SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    session = InferenceSession(str(model_path), options, providers=[provider])
    session.disable_fallback()
    return session











if __name__ == '__main__':

    # create output directory
    # timestamp = utils.get_timestamp(include_seconds=True)
    # output_directory = utils.create_directory(general.OUTPUT_PATH, name=None)

    # create logger
    logger = utils.create_logger(
        name='text_classifier',
        file_path=Path(general.OUTPUT_PATH / 'train.log'),
        level='debug',
        file_level='debug',
        console_level='error'
    )

    # load training config file
    logger.info("Loading config file.")
    config = utils.load_yaml(Path(general.INPUT_PATH / "spec.yaml"))
    dataset_info = config.get('dataset', None)
    train_info = config.get('train', None)

    # load dataset
    logger.info("Loading dataset.")
    dataset_path = str(Path(general.INPUT_PATH / dataset_info.get('file_name')))
    dataset_dict = load_dataset(
        dataset_info.get('file_type'),
        data_files=dataset_path
    )

    # converts the dataset dict to a huggingface dataset object
    dataset = dataset_dict['train']

    # apply train-test split
    logger.info(f"Applying train-test split with test size of {dataset_info.get('test_size')}")
    dataset = dataset.train_test_split(test_size=dataset_info.get('test_size'))

    # cast label column into a ClassLabel type
    class_names = dataset_info.get('label_classes')
    dataset = dataset.cast_column(
        dataset_info.get('label_column'),
        ClassLabel(num_classes=len(class_names), names=class_names)
    )

    # load the pretrained model tokenizer
    checkpoint = train_info.get('pretrained_checkpoint')
    logger.info(f"Loading pretrained Tokenizer for {checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # encode the text data with the tokenizer
    logger.info(f"Encoding text data with the tokenizer")
    encoded_text = dataset.map(
        lambda x: tokenizer(
            x[dataset_info.get('target_column')],
            padding=train_info.get('padding'),
            truncation=train_info.get('truncation')
        ),
        batched=True
    )

    # set format of the encoded text to a torch tensor
    encoded_columns = [*tokenizer.model_input_names, dataset_info.get('label_column'), dataset_info.get('target_column')]
    encoded_text.set_format(type='torch', columns=encoded_columns)

    # load autoconfig and auto model objects
    logger.info(f"Loading pretrained model and config for {checkpoint}")
    config = AutoConfig.from_pretrained(checkpoint, num_labels=len(class_names))
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, config=config).to(general.DEVICE)

    # setting training args
    logger.info(f"Setting training args with train info: {train_info}")
    batch_size = int(train_info.get('batch_size'))
    logging_steps = len(encoded_text['train']) // batch_size

    output_model_dir = utils.create_directory(general.OUTPUT_PATH, name=f"{checkpoint}-finetuned")
    output_model_checkpoints_dir = utils.create_directory(general.OUTPUT_PATH, name=f"{checkpoint}-finetuned-checkpoints")

    logger.info(f"Saving best model to {output_model_dir}")
    logger.info(f"Saving model checkpoints to {output_model_checkpoints_dir}")

    training_args = TrainingArguments(
        output_dir=output_model_checkpoints_dir,
        num_train_epochs=int(train_info.get('num_train_epochs')),
        learning_rate=float(train_info.get('learning_rate')),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=float(train_info.get('weight_decay')),
        evaluation_strategy=train_info.get('evaluation_strategy'),
        save_strategy=train_info.get('save_strategy'),
        save_total_limit=int(train_info.get('save_total_limit')),
        logging_steps=logging_steps,
        disable_tqdm=False,
        push_to_hub=train_info.get('push_to_hub'),
        log_level=train_info.get('log_level'),
        load_best_model_at_end=True
    )

    compute_metrics = train_info.get('compute_metrics')
    train_metrics = Metrics(compute_metrics)

    logger.info(f"Creating Trainer object.")
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=train_metrics,
        train_dataset=encoded_text["train"],
        eval_dataset=encoded_text["test"],
        tokenizer=tokenizer
    )

    logger.info(f"Training model for {int(train_info.get('num_train_epochs'))} epochs...")
    trainer.train()

    logger.info(f"Training finished. Saving best model.")
    trainer.save_model(output_model_dir)

    logger.info(f"Creating info directory.")
    output_info_dir = utils.create_directory(general.OUTPUT_PATH, name="info")
    model_info_dir = utils.create_directory(output_info_dir, name=f"{checkpoint}-finetuned")

    # run predictions on test data
    logger.info(f"Generating predictions on the test dataset...")
    preds_output = trainer.predict(encoded_text["test"])
    test_predictions = np.argmax(preds_output.predictions, axis=1)

    if train_info.get('save_info'):
        dataset.set_format('pandas')
        df = dataset['test'][:]
        df['predicted_label'] = test_predictions

        save_path = model_info_dir / f"test_data_with_predictions.csv"
        df.to_csv(save_path, index=False)

        logger.info(f"Saved test set predictions to csv at {save_path}.")

        labels = encoded_text['train'].features['label'].names
        y_preds = np.argmax(preds_output.predictions, axis=1)
        y_valid = np.array(encoded_text['test']['label'])

        cm_normalised = plot_confusion_matrix(y_preds, y_valid, labels)
        cm_norm_path = model_info_dir / 'confusion_matrix_normalised.png'
        cm_normalised.savefig(cm_norm_path)

        logger.info(f"Saved normalized confusion matrix set predictions at {cm_norm_path}.")

        cm_counts = plot_confusion_matrix(y_preds, y_valid, labels, normalized=None)
        cm_counts_path = model_info_dir / 'confusion_matrix_counts.png'
        cm_counts.savefig(cm_counts_path)

        logger.info(f"Saved non-normalized (counts) confusion matrix set predictions at {cm_counts_path}.")

    if train_info.get('quantize'):
        logger.info(f"Converting to ONNX and Quantizing model...")
        os.environ["OMP_NUM_THREADS"] = f"{cpu_count()}"
        os.environ["OMP_WAIT_POLICY"] = "ACTIVE"

        model = (AutoModelForSequenceClassification.from_pretrained(output_model_dir).to("cpu"))
        onnx_model_path = utils.create_directory(general.OUTPUT_PATH, name="onnx")
        onnx_model_name = onnx_model_path / "model.onnx"

        convert(
            framework="pt",
            model=model,
            tokenizer=tokenizer,
            output=onnx_model_name,
            opset=12,
            pipeline_name=dataset_info.get('type')
        )

        logger.info(f"Saved ONNX model at {onnx_model_name}")

        # load newly saved onnx model
        onnx_model = create_model_for_provider(onnx_model_name)

        # now quantize the onnx model
        # classes = dataset['test'].features['label'].names
        # pipe = OnnxPipeline(onnx_model, tokenizer, classes)
        # pipe("bank of england raises interest rates")

        quantized_onnx_model_dir = utils.create_directory(general.OUTPUT_PATH, name="quantized-onnx")
        quantized_onnx_model_name = quantized_onnx_model_dir / "model.quant.onnx"

        # save quantized model
        quantize_dynamic(onnx_model_name, quantized_onnx_model_name, weight_type=QuantType.QInt8)

        logger.info(f"Saved Quantized ONNX model at {quantized_onnx_model_name}")

        # make a copy of the tokenizer
        tokenizer.save_pretrained(quantized_onnx_model_dir)

        logger.info(f"Saved tokenizer with Quantized ONNX model.")

        # load the quantized model
        logger.info(f"Loading Quantized ONNX model pipeline...")
        onnx_quantized_model = create_model_for_provider(quantized_onnx_model_name)

        id2label = {i: label for i, label in enumerate(dataset_info.get('label_classes'))}
        return_all = dataset_info.get('return_all_labels', True)

        pipe = OnnxPipeline(
            model=onnx_quantized_model,
            tokenizer=tokenizer,
            id2label=id2label,
            return_all=return_all
        )

        if train_info.get('save_info'):
            dataset.set_format('pandas')
            text = dataset['test'][dataset_info.get('target_column')][:].tolist()

            logger.info(f"Generating test set predictions with Quantized ONNX model...")
            predictions = []
            for item in tqdm(text):
                predictions.append(pipe(item))

            predictions_df = pd.DataFrame.from_dict(predictions)
            predicted_label_names = predictions_df.idxmax(axis=1).tolist()
            predicted_label_scores = predictions_df.max(axis=1).tolist()

            label2id = {label: i for i, label in enumerate(dataset_info.get('label_classes'))}
            predicted_labels = [label2id[label] for label in predicted_label_names]

            df = dataset['test'][:]
            df['prediction'] = predictions
            df['predicted_label'] = predicted_labels
            df['predicted_label_name'] = predicted_label_names
            df['predicted_label_score'] = predicted_label_scores

            model_info_dir = utils.create_directory(output_info_dir, name="quantized-onnx")

            save_path = model_info_dir / f"test_data_with_predictions.csv"
            df.to_csv(save_path, index=False)

            logger.info(f"Saved test set predictions data at {save_path}.")

            y_valid = df['label'].tolist()

            cm_normalised = plot_confusion_matrix(predicted_labels, y_valid, class_names)
            cm_norm_path = model_info_dir / 'confusion_matrix_normalised.png'
            cm_normalised.savefig(cm_norm_path)

            logger.info(f"Saved normalized confusion matrix set predictions at {cm_norm_path}.")

            cm_counts_path = model_info_dir / 'confusion_matrix_counts.png'
            cm_counts = plot_confusion_matrix(predicted_labels, y_valid, class_names, normalized=None)
            cm_counts.savefig(cm_counts_path)

            logger.info(f"Saved non-normalized (counts) confusion matrix set predictions at {cm_counts_path}.")

    if train_info.get('delete_checkpoints_after_training'):
        logger.info(f"Deleting checkpoints created during training...")
        shutil.rmtree(output_model_checkpoints_dir)

    if train_info.get('delete_unquantized_onnx_after_training'):
        logger.info(f"Deleting ONNX model created before Quantizing...")
        shutil.rmtree(onnx_model_path)
