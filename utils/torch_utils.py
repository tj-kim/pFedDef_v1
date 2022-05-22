import warnings

import torch
import torch.nn as nn
import numpy as np
import itertools


def krum_learners(learners, target_learner, f):
    # learners = single learner from all clients
    # target_learner = hypothesis learner 
    
    target_state_dict = target_learner.model.state_dict(keep_vars=True)
    for key in target_state_dict:
#         print(key)
        if target_state_dict[key].data.dtype == torch.float32:

            distance_matrix = np.zeros([len(learners),len(learners)])
            state_dict_vals_log = []

            for learner_id, learner in enumerate(learners):
                state_dict_vals_log += [learner.model.state_dict(keep_vars=True)[key].cpu().detach().numpy()]

            for i1,i2 in itertools.product(range(len(learners)),range(len(learners))):
                if i1 != i2 and distance_matrix[i1, i2] == 0:
                    c = state_dict_vals_log[i1]- state_dict_vals_log[i2]
                    distance_matrix[i1, i2] = np.linalg.norm(c)
                    distance_matrix[i2, i1] = np.linalg.norm(c)

            # from distance matrix calculate mean for n-f-2, n=num learners, f = num_sybl, 2
            krum_vector = np.zeros(len(learners))
            # value of k
            k = len(learners) - f - 2  

            # using np.argpartition()
            for i in range(len(learners)):
                result = np.argpartition(distance_matrix[i], k)
                krum_vector[i] = np.sum(distance_matrix[i][result[:k]])

            # Update weight using krum 
            min_idx = np.argmin(krum_vector)
            target_state_dict[key].data = learners[min_idx].model.state_dict(keep_vars=True)[key].data.clone()

        else:
            # tracked batches
            target_state_dict[key].data.fill_(0)
            for learner_id, learner in enumerate(learners):
                state_dict = learner.model.state_dict()
                target_state_dict[key].data += state_dict[key].data.clone()
        
    return 

def average_learners(
        learners,
        target_learner,
        weights=None,
        average_params=True,
        average_gradients=False):
    """
    Compute the average of a list of learners_ensemble and store it into learner

    :param learners:
    :type learners: List[Learner]
    :param target_learner:
    :type target_learner: Learner
    :param weights: tensor of the same size as learners_ensemble, having values between 0 and 1, and summing to 1,
                    if None, uniform learners_weights are used
    :param average_params: if set to true the parameters are averaged; default is True
    :param average_gradients: if set to true the gradient are also averaged; default is False
    :type weights: torch.Tensor

    """
    if not average_params and not average_gradients:
        return

    if weights is None:
        n_learners = len(learners)
        weights = (1 / n_learners) * torch.ones(n_learners, device=learners[0].device)

    else:
        weights = weights.to(learners[0].device)

    target_state_dict = target_learner.model.state_dict(keep_vars=True)

    for key in target_state_dict:

        if target_state_dict[key].data.dtype == torch.float32:

            if average_params:
                target_state_dict[key].data.fill_(0.)

            if average_gradients:
                target_state_dict[key].grad = target_state_dict[key].data.clone()
                target_state_dict[key].grad.data.fill_(0.)

            for learner_id, learner in enumerate(learners):
                state_dict = learner.model.state_dict(keep_vars=True)

                if average_params:
                    target_state_dict[key].data += weights[learner_id] * state_dict[key].data.clone()

                if average_gradients:
                    if state_dict[key].grad is not None:
                        target_state_dict[key].grad += weights[learner_id] * state_dict[key].grad.clone()
                    elif state_dict[key].requires_grad:
                        warnings.warn(
                            "trying to average_gradients before back propagation,"
                            " you should set `average_gradients=False`."
                        )

        else:
            # tracked batches
            target_state_dict[key].data.fill_(0)
            for learner_id, learner in enumerate(learners):
                state_dict = learner.model.state_dict()
                target_state_dict[key].data += state_dict[key].data.clone()


def partial_average(learners, average_learner, alpha):
    """
    performs a step towards aggregation for learners, i.e.

    .. math::
        \forall i,~x_{i}^{k+1} = (1-\alpha) x_{i}^{k} + \alpha \bar{x}^{k}

    :param learners:
    :type learners: List[Learner]
    :param average_learner:
    :type average_learner: Learner
    :param alpha:  expected to be in the range [0, 1]
    :type: float

    """
    source_state_dict = average_learner.model.state_dict()

    target_state_dicts = [learner.model.state_dict() for learner in learners]

    for key in source_state_dict:
        if source_state_dict[key].data.dtype == torch.float32:
            for target_state_dict in target_state_dicts:
                target_state_dict[key].data =\
                    (1-alpha) * target_state_dict[key].data + alpha * source_state_dict[key].data


def differentiate_learner(target, reference_state_dict, coeff=1.):
    """
    set the gradient of the model to be the difference between `target` and `reference` multiplied by `coeff`

    :param target:
    :type target: Learner
    :param reference_state_dict:
    :type reference_state_dict: OrderedDict[str, Tensor]
    :param coeff: default is 1.
    :type: float

    """
    target_state_dict = target.model.state_dict(keep_vars=True)

    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:

            target_state_dict[key].grad = \
                coeff * (target_state_dict[key].data.clone() - reference_state_dict[key].data.clone())


def copy_model(target, source):
    """
    Copy learners_weights from target to source
    :param target:
    :type target: nn.Module
    :param source:
    :type source: nn.Module
    :return: None

    """
    target.load_state_dict(source.state_dict())


def simplex_projection(v, s=1):
    """
    Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):

    .. math::
        min_w 0.5 * || w - v ||_2^2,~s.t. \sum_i w_i = s, w_i >= 0

    Parameters
    ----------
    v: (n,) torch tensor,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex

    Returns
    -------
    w: (n,) torch tensor,
       Euclidean projection of v on the simplex

    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.

    References
    ----------
    [1] Wang, Weiran, and Miguel A. Carreira-Perpinán. "Projection
        onto the probability simplex: An efficient algorithm with a
        simple proof, and an application." arXiv preprint
        arXiv:1309.1541 (2013)
        https://arxiv.org/pdf/1309.1541.pdf

    """

    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape

    u, _ = torch.sort(v, descending=True)

    cssv = torch.cumsum(u, dim=0)

    rho = int(torch.nonzero(u * torch.arange(1, n + 1) > (cssv - s))[-1][0])

    lambda_ = - float(cssv[rho] - s) / (1 + rho)

    w = v + lambda_

    w = (w * (w > 0)).clip(min=0)

    return w


