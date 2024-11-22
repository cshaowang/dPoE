import torch
from modules.dPoeModel import DPoE


def mv_model_loader(path='./sample_model.pt', img_size=(1, 32, 32), view_num=2, latent_dims=None,
                    hid_dim=256, tc_dim=1000, share=False, tau=.67, poe=True):
    """
    Loads a trained model.

    Parameters
    ----------
    path : string
        Path to folder where model is saved. For example
        './models/model_name.pt'.

    img_size : cell
        input data size
    view_num : int
        number of views
    latent_dims : dict
        view-common and view-specific latent representation dimensions
    hid_dim : int
        dimension of hidden representation
    tc_dim : int
        dimension of TC discriminator layer
    share : bool
    tau : float
    poe : bool
        Using Product-of-Experts or Mixture-of-Experts

    """
    if latent_dims is None:
        latent_dims = {"comm": 10, "spec": 10}
    path_to_model = path
    # Load the model
    model = DPoE(img_size=img_size, view_num=view_num, latent_dims=latent_dims, hidden_dim=hid_dim, tc_dim=tc_dim,
                 share=share, tau=tau, poe=poe)
    if torch.cuda.is_available():
        # Load all tensors onto GPU: map_location=lambda storage, loc: storage.cuda()
        model.load_state_dict(torch.load(path_to_model, map_location=lambda storage, loc: storage.cuda()))
        model.cuda()
        print("Cuda is available.")
    else:
        # Load all tensors onto the CPU: map_location=lambda storage, loc: storage or map_location=torch.device('cpu')
        model.load_state_dict(torch.load(path_to_model, map_location=lambda storage, loc: storage))
        print("No cuda available!")

    return model
