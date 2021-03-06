# Auto NLP (with Huggingface Transformers)

Work in progress. Currently supports training models for single/multiclass text classification and multilabel text 
classification.

### Features:
- Saves the best Pytorch model at the end of training.
- Converts Pytorch model to ONNX for production.
- Quantizes ONNX model for reduced file size.
- Outputs confusion matrices and test set predictions to evaluate performance.

### To use:

1. Place your training dataset into the input folder along with a spec YAML file containing information about the name of 
the text field, name of the label fields, any training parameter information, etc.  
2. Run `train.py` in the src directory.
3. The trained models will be saved in the output directory.  

### Installing:

Install dependencies:   
```bash
pip install -r requirements.txt
```

Install Pytorch:  
```bash
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio===0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

### Notes
- For multilabel classification, I have only managed to get it to work if the dataset is saved as a JSON file and the 
labels are prepared with Scikit-Learn's MultiLabelBinarizer. The labels inside the list also need to be converted to 
floats rather than ints.


### To do:
- Add support for token classification
- Add distillation
- Add hyperparameter optimization

