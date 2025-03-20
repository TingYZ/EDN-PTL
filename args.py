# Network Arguments
# 使用两模型不同之处：1.导入模型，3.下面args['dyn_size']=64 or 224
args = {}
args['encoder_size'] = 64
args['decoder_size'] = 128
args['in_length'] = 16
args['out_length'] = 25
args['grid_size'] = (13,3)
args['soc_conv_depth'] = 64
args['conv_3x1_depth'] = 16
args['dyn_embedding_size'] = 32
args['input_embedding_size'] = 32
args['trained_model'] = 'trained_models/'
args['batch_size'] = 128

args['dyn_size'] = 64  # 时序信息长度
args['model_type'] = 'mine'