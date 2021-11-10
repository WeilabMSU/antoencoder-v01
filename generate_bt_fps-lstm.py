from fairseq.models.lstm import LSTMModel
import argparse
import sys
import numpy as np
import torch
from omegaconf import DictConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf


def load_pretrain_model(model_name_or_path, checkpoint_file, data_name_or_path, tokenizer):
    '''Currently only load to cpu()'''

    # load model
    pretrain_model = LSTMModel.from_pretrained(
        model_name_or_path,
        checkpoint_file,
        data_name_or_path,  # dict_dir,
        tokenizer=tokenizer,
    )
    pretrain_model.eval()
    return pretrain_model


def extract_hidden(pretrain_model, target_file):

    sample_num = 0
    for i, line in enumerate(open(target_file)):
        if len(line.strip()) == 0:
            continue
        sample_num += 1
    hidden_features = []

    for i, line in enumerate(open(target_file)):
        if len(line.strip()) == 0:
            continue
        tokens = pretrain_model.encode(line.strip())
#        print(pretrain_model.max_positions)
        print(tokens)
        if len(tokens) > pretrain_model.max_positions[0]:
            tokens = torch.cat(
                (tokens[:pretrain_model.max_positions[0] - 1], tokens[-1].unsqueeze(0)))
        
        lth=torch.tensor(list(tokens.shape))
        tokens=torch.reshape(tokens, (1,-1))
        
        all_layer_hiddens = pretrain_model.models[0].encoder(tokens,lth)
#        print(all_layer_hiddens[1].shape)

#        hidden_info = all_layer_hiddens['inner_states'][-1]
        # last_hidden shape [tokens_num, sample_num(default=1), hidden_dim]

        # hidden_features.append(hidden_info.squeeze(1).cpu().detach().numpy())
        hidden_features.append(all_layer_hiddens[1].cpu().detach().numpy().reshape(-1))
#        print(hidden_features[i])
    
    hidden_features=np.array(hidden_features)
    # hidden_features type: dict, length: samples_num
    return hidden_features




def main(args):
    pretrain_model = load_pretrain_model(
        args.model_name_or_path, args.checkpoint_file, args.data_name_or_path, args.tokenizer)

    hidden_info = extract_hidden(pretrain_model, args.target_file)
#    print(hidden_info)

    print('Generate features from hidden information')
    print(f'Features shape: {np.shape(hidden_info)}')
    np.save(args.save_feature_path, hidden_info)


def parse_args(args):
    parser = argparse.ArgumentParser(description="Tools kit for downstream jobs")

    parser.add_argument('--model_name_or_path', default=None, type=str,
                        help='Pretrained model folder')
    parser.add_argument('--checkpoint_file', default='checkpoint_best.pt', type=str,
                        help='Pretrained model name')
    parser.add_argument('--data_name_or_path', default=None, type=str,
                        help="Pre-training dataset folder")
#    parser.add_argument('--smi_voc', default='example-smiles/dict.txt', type=str,
#                        help="Pre-training dict filename(full path)")
    parser.add_argument('--tokenizer', default='smi', type=str)
    parser.add_argument('--target_file', default=None, type=str,
                        help="Target file for feature extraction, default format is .smi")
    parser.add_argument('--save_feature_path', default='extract_f1.npy', type=str,
                        help="Saving feature filename(path)")
    args = parser.parse_args()
    return args


def cli_main():
    args = parse_args(sys.argv[1:])
#    cfg = convert_namespace_to_omegaconf(args)
    main(args)


if __name__ == '__main__':
    cli_main()
    print('End!')
    
