# DeepTUL
PyTorch implementation of AAMAS'20  paper-**Trajectory-User Linking with Attentive Recurrent Network**

# Datasets
The sample data to evaluate our model can be found in the data folder, which contains 200+ users and ready for directly used. 

# Requirements
- Python 2.7
- [Pytorch](https://pytorch.org/previous-versions/) 0.20

cPickle is used in the project to store the preprocessed data and parameters. While appearing some warnings, pytorch 0.3.0 can also be used.

# Project Structure
- /codes
    - [main.py](https://github.com/CodyMiao/DeepTUL/blob/master/codes/main.py)
    - [model.py](https://github.com/CodyMiao/DeepTUL/blob/master/codes/model.py) # define models
    - [masked_cross_entropy.py](https://github.com/CodyMiao/DeepTUL/blob/master/codes/masked_cross_entropy.py) #calculate entropy
    - [train.py](https://github.com/CodyMiao/DeepTUL/blob/master/codes/train.py)  # define tools for train the model
    - /data
- /data # preprocessed foursquare sample data (pickle file)
- /docs # paper and presentation file
- /resutls # the default save path when training the model

# Usage
Train a new model:

> ```python
> python main.py 
> ```

Other parameters (refer to [main.py](https://github.com/CodyMiao/DeepTUL/blob/master/codes/main.py)):
- for training: 
    - learning_rate, lr_step, lr_decay, L2, clip, epoch_max, dropout_p
- model definition: 
    - loc_emb_size, uid_emb_size, tim_emb_size, hidden_size, rnn_type, attn_type
    - strategies_type: AVE-sdot,AVE-dot,MAX-sdot,MAX-dot

# Other

More specific data and data processing methods will be given later