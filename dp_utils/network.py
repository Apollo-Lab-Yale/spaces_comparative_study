#####################################
#@markdown ### **Network**
from dp_utils.vision_encoder import get_resnet, replace_bn_with_gn
from dp_utils.unet import ConditionalUnet1D
from torch import nn

def create_nets(vision_feature_dim, lowdim_obs_dim, action_dim, obs_horizon):
    # construct ResNet18 encoder
    # if you have multiple camera views, use seperate encoder weights for each view.
    vision_encoder = get_resnet('resnet18')

    # IMPORTANT!
    # replace all BatchNorm with GroupNorm to work with EMA
    # performance will tank if you forget to do this!
    vision_encoder = replace_bn_with_gn(vision_encoder)
    # observation feature has 514 dims in total per step

    obs_dim = vision_feature_dim + lowdim_obs_dim
    # Print OBS_DIM to check the total observation dimension
    print(f"OBS_DIM: {obs_dim}")
    # create network object
    print(f"GLOBALCONDDIM: ", obs_dim*obs_horizon)
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    )

    # the final arch has 2 parts
    nets = nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'noise_pred_net': noise_pred_net
    })

    return nets