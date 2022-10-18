import torch
import os
import pickle
import numpy as np


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class TrainUtil():
    def __init__(self, checkpoint_path='checkpoints', version='mcts_nas_net_v1'):
        self.checkpoint_path = checkpoint_path
        self.version = version
        return

    def save_model_fixed(self, epoch, fixed_cnn, fixed_cnn_optmizer, save_best=False, **kwargs):
        filename = os.path.join(self.checkpoint_path, self.version) + '.pth'
        # Torch Save State Dict
        state = {
            'epoch': epoch+1,
            'shared_cnn': fixed_cnn.state_dict(),
            'shared_cnn_optmizer': fixed_cnn_optmizer.state_dict(),
        }
        for key, value in kwargs.items():
            state[key] = value
        torch.save(state, filename)
        filename = os.path.join(self.checkpoint_path, self.version) + '_best.pth'
        if save_best:
            torch.save(state, filename)
        return

    def load_model_fixed(self, fixed_cnn, fixed_cnn_optmizer, **kwargs):
        filename = os.path.join(self.checkpoint_path, self.version) + '.pth'
        # Load Torch State Dict
        checkpoints = torch.load(filename)
        fixed_cnn.load_state_dict(checkpoints['fixed_cnn'])
        fixed_cnn_optmizer.load_state_dict(checkpoints['fixed_cnn_optmizer'])
        print(filename + " Loaded!")
        return checkpoints

    def save_model(self,
                   mcts,
                   shared_cnn,
                   shared_cnn_optmizer,
                   shared_cnn_schduler,
                   estimator,
                   estimator_optmizer,
                   epoch,
                   **kwargs):
        mcts_filename = os.path.join(self.checkpoint_path, self.version) + '_mcts' + '.pkl'
        filename = os.path.join(self.checkpoint_path, self.version) + '.pth'

        # Torch Save State Dict
        state = {
            'epoch': epoch+1,
            'shared_cnn': shared_cnn.state_dict(),
            'shared_cnn_optmizer': shared_cnn_optmizer.state_dict(),
            'shared_cnn_schduler': shared_cnn_schduler.state_dict(),
            'estimator': estimator.state_dict(),
            'estimator_optmizer': estimator_optmizer.state_dict()
        }
        for key, value in kwargs.items():
            state[key] = value
        torch.save(state, filename)
        print(filename + " saved!")

        # Save MCTS to pickle
        rolloutPolicy, searchPolicy = mcts.rollout, mcts.searchPolicy
        mcts.rollout, mcts.searchPolicy = None, None
        with open(mcts_filename, 'wb') as handle:
            pickle.dump(mcts, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(mcts_filename + " Saved!")
        mcts.rollout, mcts.searchPolicy = rolloutPolicy, searchPolicy
        return

    def load_model(self,
                   shared_cnn,
                   shared_cnn_optmizer,
                   shared_cnn_schduler,
                   estimator,
                   estimator_optmizer,
                   **kwargs):

        filename = os.path.join(self.checkpoint_path, self.version) + '.pth'
        mcts_filename = os.path.join(self.checkpoint_path, self.version) + '_mcts' + '.pkl'

        # Load Torch State Dict
        checkpoints = torch.load(filename)
        shared_cnn.load_state_dict(checkpoints['shared_cnn'])
        shared_cnn_optmizer.load_state_dict(checkpoints['shared_cnn_optmizer'])
        shared_cnn_schduler.load_state_dict(checkpoints['shared_cnn_schduler'])
        shared_cnn_schduler.optimizer = shared_cnn_optmizer
        estimator.load_state_dict(checkpoints['estimator'])
        estimator_optmizer.load_state_dict(checkpoints['estimator_optmizer'])
        print(filename + " Loaded!")

        # Load MCTS
        with open(mcts_filename, 'rb') as handle:
            mcts = pickle.load(handle)
        print(mcts_filename + " Loaded!")
        return checkpoints, mcts





def get_wieghts_of_clean_noisy(args, loss, targets, u, lamda, noisy_index):

    loss = torch.tensor(loss)
    exploss = torch.exp(loss / lamda)

    # print(exploss, u)
    p = exploss / torch.sum(exploss)
    p = p.detach()

    # print(">>>>>: ", p[0], targets, " :<<<<<<<")

    print(1/len(loss))
    cls_p = []
    bool_index = np.array([False] * len(targets))
    bool_index[noisy_index] = True
    noisy_weight_mean = p[bool_index].mean().item()
    clean_weight_mean = p[~bool_index].mean().item()
    cls_p.append(clean_weight_mean)
    cls_p.append(noisy_weight_mean)


    clean_top_k, _ =  torch.topk(loss[~bool_index],len(loss[~bool_index])//5)
    noisy_top_k, _ = torch.topk(loss[bool_index], len(loss[~bool_index])//5)
    print( 'clean samples average loss:', loss[~bool_index].mean().item(), clean_top_k.mean().item())
    print('noisy samples average loss:', loss[bool_index].mean().item(), noisy_top_k.mean().item())



    return cls_p


