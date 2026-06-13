# Adaptive Feature Interaction and Edge Refinement Networks for Camouflaged Object Detection

# 1 Downloading necessary data:
- downloading testing dataset and move it into ./CodDataset/TestDataset/, which can be found in this download link (https://pan.baidu.com/s/1RfNy0hPcGAvJyeUeE07nKA?pwd=1234 Code: 1234).
- downloading training dataset and move it into ./CodDataset/TrainDataset/, which can be found in this download link (https://pan.baidu.com/s/1RfNy0hPcGAvJyeUeE07nKA?pwd=1234 Code: 1234).
- downloading smt pretrained weights and move it into ./smt_tiny.pth, which can be found in this download link (https://pan.baidu.com/s/1RfNy0hPcGAvJyeUeE07nKA?pwd=1234 Code: 1234).

# 2 Training Configuration:
- Set parameters and the save path (./out/AFNet/) in the options_cod.py file, then run the train.py file.

# 3 Testing Configuration:
- After you download all the pre-trained model and testing dataset, just run test.py to generate the final prediction map.

# 4 Evaluating your trained model:
Assigning your costumed path, like method, mask_root and pred_root in eval.py. Just run eval.py to evaluate the trained model.

# 5 Contact
Feel free to send e-mails to me (1916342521@qq.com).<br>


