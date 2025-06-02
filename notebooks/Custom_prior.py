import torch
from scipy import stats
import numpy as np


def get_prior(path_to_prior):
    params_dict = {}
    param_names = []
    lower_bounds = []
    upper_bounds = []
    with open(path_to_prior, "r") as f:
        lines = f.readlines()
        for line in lines:
            line_split = line.split()
            param_name = line_split[0]
            dist_type = line_split[1]
            dist_pars = line_split[2:]
            params_dict[param_name] = [dist_type, [float(p) for p in dist_pars]]
            param_names.append(param_name)
            if dist_type == "u":
                lower_bounds.append(float(dist_pars[0]))
                upper_bounds.append(float(dist_pars[1]))
            elif dist_type == "n":
                lower_bounds.append(-float("inf"))
                upper_bounds.append(float("inf"))
            elif dist_type == "tn":
                lower_bounds.append(float(dist_pars[0]))
                upper_bounds.append(float(dist_pars[1]))
            else:
                raise NotImplementedError

    lower_bounds = torch.as_tensor(lower_bounds)
    upper_bounds = torch.as_tensor(upper_bounds)

    return params_dict, param_names, lower_bounds, upper_bounds


class CustomPrior:
    def __init__(self, params_dict, names, return_numpy: bool = False, device=None):
        self.return_numpy = return_numpy
        self.dist_list = []
        for name in names:
            dist_type, dist_pars = params_dict[name]
            if dist_type == "u":
                self.dist_list.append(
                    stats.uniform(loc=dist_pars[0], scale=dist_pars[1] - dist_pars[0])
                )
            elif dist_type == "n":
                self.dist_list.append(
                    stats.norm(loc=dist_pars[0], scale=dist_pars[1])
                )
            elif dist_type == "tn":
                self.dist_list.append(stats.truncnorm(
                    a=(dist_pars[0] - dist_pars[2]) / dist_pars[3],
                    b=(dist_pars[1] - dist_pars[2]) / dist_pars[3],
                    loc=dist_pars[2],
                    scale=dist_pars[3],
                ))

        self.mean = torch.as_tensor([dist.mean() for dist in self.dist_list])
        self.variance = torch.as_tensor([dist.var() for dist in self.dist_list])
        self.device = device


    def sample(self, sample_shape=torch.Size([])):
        samples = []
        for dist in self.dist_list:
            sample = dist.rvs(size=sample_shape)
            samples.append(sample)
        result = np.stack(samples, axis=-1)
        if self.return_numpy:
            return result
        else:
            if self.device is None:
                return torch.as_tensor(result)
            else:
                return torch.as_tensor(result).to(self.device)

    def log_prob(self, x):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        log_probs = []
        for i, dist in enumerate(self.dist_list):
            xi = x[..., i]
            log_probs.append(dist.logpdf(xi))
        total_log_prob = np.sum(np.stack(log_probs, axis=-1), axis=-1)
        if self.return_numpy:
            return total_log_prob
        else:
            if self.device is None:
                return torch.as_tensor(total_log_prob)
            else:
                return torch.as_tensor(total_log_prob).to(self.device)