# main python imports
import os
import shutil
from pathlib import Path

# local imports
import src.classes as classes
from src.settings import general
import src.utils as utils

# huggingface imports
from datasets import load_dataset, ClassLabel
from transformers.convert_graph_to_onnx import convert
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, DataCollatorWithPadding, Trainer, TrainingArguments
)

# other package imports
import numpy as np
import pandas as pd
from tqdm import tqdm
from psutil import cpu_count
from onnxruntime.quantization import quantize_dynamic, QuantType


if __name__ == '__main__':
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

    problem_type = dataset_info.get('problem_type')

    # load dataset
    logger.info("Loading dataset.")
    dataset_path = str(Path(general.INPUT_PATH / dataset_info.get('file_name')))

    dataset_dict = load_dataset(
        dataset_info.get('file_type'),
        data_files=dataset_path,
    )

    # converts the dataset dict to a huggingface dataset object
    dataset = dataset_dict['train']
    class_names = dataset_info.get('label_classes')

    # apply train-test split
    logger.info(f"Applying train-test split with test size of {dataset_info.get('test_size')}")
    dataset = dataset.train_test_split(test_size=dataset_info.get('test_size'))

    # cast label column into a ClassLabel type
    if problem_type != 'multi_label_classification':
        dataset = dataset.cast_column(
            dataset_info.get('label_column'),
            ClassLabel(num_classes=len(class_names), names=class_names)
        )

    # load the pretrained model tokenizer
    checkpoint = train_info.get('pretrained_checkpoint')
    logger.info(f"Loading pretrained Tokenizer for {checkpoint}")

    auto_config = AutoConfig.from_pretrained(
        checkpoint,
        problem_type=problem_type,
        num_labels=len(class_names)
    )

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        config=auto_config
    )

    # encode the text data with the tokenizer
    logger.info(f"Encoding text data with the tokenizer")


    if problem_type == 'multi_label_classification':
        encoded_text_train = tokenizer(
            dataset['train']['text'],
            truncation=True,
            padding=True,
            return_tensors='pt'
        )

        encoded_text_test = tokenizer(
            dataset['test']['text'],
            truncation=True,
            padding=True,
            return_tensors='pt'
        )

        train_dataset = classes.MultiLabelDataset(encoded_text_train, dataset['train']['labels'])
        test_dataset = classes.MultiLabelDataset(encoded_text_test, dataset['test']['labels'])
    else:
        encoded_text = dataset.map(
            lambda x: tokenizer(
                x[dataset_info.get('target_column')],
                padding=train_info.get('padding'),
                truncation=train_info.get('truncation'),
            ),
            batched=True
        )

        encoded_columns = [*tokenizer.model_input_names, dataset_info.get('label_column'),
                           dataset_info.get('target_column')]

        # set format of the encoded text to a torch tensor
        encoded_text.set_format(type='pt', columns=encoded_columns)

        train_dataset = encoded_text["train"]
        test_dataset = encoded_text["test"]


    # load autoconfig and auto model objects
    logger.info(f"Loading pretrained model and config for {checkpoint}")
    # config = AutoConfig.from_pretrained(checkpoint, problem_type=problem_type, num_labels=len(class_names))
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        config=auto_config,
        # problem_type=problem_type,
        # num_labels=len(class_names)
    ).to(general.DEVICE)

    # setting training args
    logger.info(f"Setting training args with train info: {train_info}")
    batch_size = int(train_info.get('batch_size'))
    logging_steps = len(train_dataset) // batch_size

    output_model_dir = utils.create_directory(general.OUTPUT_PATH, name=f"{checkpoint}-finetuned")
    output_model_checkpoints_dir = utils.create_directory(general.OUTPUT_PATH, name=f"{checkpoint}-finetuned-checkpoints")

    logger.info(f"Saving best model to {output_model_dir}")
    logger.info(f"Saving model checkpoints to {output_model_checkpoints_dir}")

    training_args = TrainingArguments(
        output_dir=output_model_checkpoints_dir,
        num_train_epochs=int(train_info.get('num_train_epochs')),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy=train_info.get('evaluation_strategy'),
        save_strategy=train_info.get('save_strategy'),
        save_total_limit=int(train_info.get('save_total_limit')),
        logging_steps=logging_steps,
        disable_tqdm=False,
        push_to_hub=train_info.get('push_to_hub'),
        log_level=train_info.get('log_level'),
        load_best_model_at_end=True
    )

    learning_rate = train_info.get('learning_rate', None)
    if learning_rate is not None:
        training_args.learning_rate = float(learning_rate)

    weight_decay = train_info.get('weight_decay', None)
    if weight_decay is not None:
        training_args.weight_decay = float(weight_decay)

    logger.info(f"Creating Trainer object.")

    if problem_type == 'multi_label_classification':
        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer
        )

        train_metrics = None
    else:
        data_collator = None
        compute_metrics = train_info.get('compute_metrics')
        train_metrics = classes.Metrics(compute_metrics)

    trainer = Trainer(
        model=model,
        args=training_args,
        # compute_metrics=train_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    if train_metrics is not None:
        trainer.compute_metrics = train_metrics

    logger.info(f"Training model for {int(train_info.get('num_train_epochs'))} epochs...")
    trainer.train()

    logger.info(f"Training finished. Saving best model.")
    trainer.save_model(output_model_dir)

    logger.info(f"Creating info directory.")
    output_info_dir = utils.create_directory(general.OUTPUT_PATH, name="info")
    model_info_dir = utils.create_directory(output_info_dir, name=f"{checkpoint}-finetuned")

    # run predictions on test data
    logger.info(f"Generating predictions on the test dataset...")
    preds_output = trainer.predict(test_dataset)

    if problem_type == 'multi_label_classification':
        test_predictions = utils.multilabel_preds(preds_output.predictions)
    else:
        # this needs to change for multilabel...
        test_predictions = np.argmax(preds_output.predictions, axis=1)

    if train_info.get('save_info'):
        dataset.set_format('pandas')
        df = dataset['test'][:]
        df['predicted_label'] = test_predictions

        save_path = model_info_dir / f"test_data_with_predictions.csv"
        df.to_csv(save_path, index=False)

        logger.info(f"Saved test set predictions to csv at {save_path}.")

        # labels = class_names
        if problem_type == 'multi_label_classification':
            y_preds_ml = [[(1 if p >= 0.5 else 0) for p in item] for item in test_predictions]
            y_valid_ml = [[(1 if p >= 0.5 else 0) for p in item] for item in dataset['test']['labels']]

            cms = utils.plot_multilabel_confusion_matrix(y_true=y_valid_ml, y_pred=y_preds_ml, class_names=class_names)

            for cm in cms:
                cm_save_path = model_info_dir / f"confusion_matrix_{cm[0]}.jpg"
                cm[1].savefig(cm_save_path)

            logger.info(f"Saved multilabel confusion matrices set predictions at {model_info_dir}.")
        else:
            y_preds = np.argmax(preds_output.predictions, axis=1)
            y_valid = np.array(test_dataset[dataset_info.get('label_column')])
            cm_normalised = utils.plot_confusion_matrix(y_preds, y_valid, class_names)
            cm_norm_path = model_info_dir / 'confusion_matrix_normalised.png'
            cm_normalised.savefig(cm_norm_path)

            logger.info(f"Saved normalized confusion matrix set predictions at {cm_norm_path}.")

            cm_counts = utils.plot_confusion_matrix(y_preds, y_valid, class_names, normalized=None)
            cm_counts_path = model_info_dir / 'confusion_matrix_counts.png'
            cm_counts.savefig(cm_counts_path)

            logger.info(f"Saved non-normalized (counts) confusion matrix set predictions at {cm_counts_path}.")

    if train_info.get('quantize'):
        logger.info(f"Converting to ONNX and Quantizing model...")
        os.environ["OMP_NUM_THREADS"] = f"{cpu_count()}"
        os.environ["OMP_WAIT_POLICY"] = "ACTIVE"

        model = AutoModelForSequenceClassification.from_pretrained(
            output_model_dir,
            problem_type=problem_type,
            num_labels=len(class_names)
        ).to("cpu")

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
        onnx_model = utils.create_model_for_provider(onnx_model_name)

        # set directory and file name for quantized onnx model
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
        onnx_quantized_model = utils.create_model_for_provider(quantized_onnx_model_name)

        id2label = {i: label for i, label in enumerate(dataset_info.get('label_classes'))}
        return_all = dataset_info.get('return_all_labels', True)

        pipe = classes.OnnxPipeline(
            model=onnx_quantized_model,
            tokenizer=tokenizer,
            id2label=id2label,
            return_all=return_all,
            problem_type=problem_type
        )

        # preds = pipe("consumer spending falls after cost of living crisis deepens")
        if train_info.get('save_info'):
            dataset.set_format('pandas')
            text = dataset['test'][dataset_info.get('target_column')][:].tolist()

            logger.info(f"Generating test set predictions with Quantized ONNX model...")
            predictions = []
            for item in tqdm(text):
                predictions.append(pipe(item))

            predictions_df = pd.DataFrame(predictions)

            if problem_type == 'multi_label_classification':
                predicted_labels = [[(1 if score >= 0.5 else 0) for label, score in item.items()] for item in predictions]
                predicted_label_names = [', '.join([label for label, score in item.items() if score >= 0.5]).strip() for item in predictions]
                predicted_label_scores = [[score for label, score in item.items()] for item in predictions]
            else:
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

            if problem_type == 'multi_label_classification':
                y_preds_ml = [[(1 if p >= 0.5 else 0) for p in item] for item in predicted_label_scores]
                y_valid_ml = [[(1 if p >= 0.5 else 0) for p in item] for item in dataset['test']['labels']]

                cms = utils.plot_multilabel_confusion_matrix(y_true=y_valid_ml, y_pred=y_preds_ml, class_names=class_names)

                for cm in cms:
                    cm_save_path = model_info_dir / f"confusion_matrix_{cm[0]}.jpg"
                    cm[1].savefig(cm_save_path)

                logger.info(f"Saved multilabel confusion matrices set predictions at {model_info_dir}.")

            else:
                y_valid = df[dataset_info.get('label_column')].tolist()

                cm_normalised = utils.plot_confusion_matrix(predicted_labels, y_valid, class_names)
                cm_norm_path = model_info_dir / 'confusion_matrix_normalised.png'
                cm_normalised.savefig(cm_norm_path)

                logger.info(f"Saved normalized confusion matrix set predictions at {cm_norm_path}.")

                cm_counts_path = model_info_dir / 'confusion_matrix_counts.png'
                cm_counts = utils.plot_confusion_matrix(predicted_labels, y_valid, class_names, normalized=None)
                cm_counts.savefig(cm_counts_path)

                logger.info(f"Saved non-normalized (counts) confusion matrix set predictions at {cm_counts_path}.")

    if train_info.get('delete_checkpoints_after_training'):
        logger.info(f"Deleting checkpoints created during training...")
        shutil.rmtree(output_model_checkpoints_dir)

    if train_info.get('delete_unquantized_onnx_after_training'):
        logger.info(f"Deleting ONNX model created before Quantizing...")
        shutil.rmtree(onnx_model_path)
