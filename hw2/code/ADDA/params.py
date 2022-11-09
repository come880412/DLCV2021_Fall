"""Params for ADDA."""

# params for dataset and data loader
data_root = "data"
dataset_mean_value = 0.5
dataset_std_value = 0.5
dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)
batch_size = 64
image_size = 28
n_cpu = 0

# params for source dataset
src_dataset = "../../hw2_data/digits/usps"  # mnistm/svhn/usps
src_csv = './data_split/usps'  # mnistm/svhn/usps

src_encoder_restore = ""
src_classifier_restore = ""
src_model_trained = True

# params for target dataset
tgt_dataset = "../../hw2_data/digits/svhn"  # usps/mnistm/svhn
tgt_csv = './data_split/svhn' # usps/mnistm/svhn
tgt_encoder_restore = ""
tgt_model_trained = True

# params for setting up models
model_root = "checkpoints_usps_svhn" # 
d_input_dims = 500
d_hidden_dims = 500
d_output_dims = 2
d_model_restore = ""

# params for training network
num_gpu = 1
num_epochs_pre = 100
log_step_pre = 20
eval_step_pre = 20
save_step_pre = 100
num_epochs = 150
log_step = 100
save_step = 100
manual_seed = None

# params for optimizing models
d_learning_rate = 5e-2
c_learning_rate = 7e-6
beta1 = 0.5
beta2 = 0.9
