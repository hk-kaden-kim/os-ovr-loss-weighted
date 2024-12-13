import torch
from torch.nn import functional as F
from .. import tools
from .. import dataset

class ovr_loss:

    def __init__(self, num_of_classes=10, mode=None, training_data=None, is_verbose=True):
        print("\n↓↓↓ Loss setup ↓↓↓")
        print(f"{self.__class__.__name__} Loaded!")
        if mode: print(f"Mode : {mode}")

        self.num_of_classes = num_of_classes
        self.mode = mode

        if self.mode:
            self.mode_namespace = list(self.mode.dict().keys())
            if 'C' in self.mode_namespace:
                need_init = self.mode.C == 'global'
            else:
                need_init = False
            self.osovr_weight = ovr_loss_weight(need_init = need_init,
                                                  num_of_classes = self.num_of_classes,
                                                  training_data = training_data,
                                                  is_verbose=is_verbose)
        
    @tools.loss_reducer
    def __call__(self, logit_values, target_labels):
        
        # One-hot encoding and Get Probability score
        enc_target_labels = tools.target_encoding(target_labels, self.num_of_classes)
        probs = F.sigmoid(logit_values)

        # Weighting for balancing
        if self.mode:
            weight_total = tools.device(torch.ones(enc_target_labels.shape))
            if 'C' in self.mode_namespace:
                weight = self.osovr_weight.get_c_weight(target_labels, enc_target_labels, 
                                                        from_global = self.mode.C == 'global')
                weight_total = weight_total * weight
            if 'F' in self.mode_namespace:
                weight = self.osovr_weight.get_f_weight(probs, enc_target_labels,
                                                        gamma = self.mode.F)
                weight_total = weight_total * weight
            if 'M' in self.mode_namespace:
                weight = self.osovr_weight.get_h_weight(probs, target_labels, enc_target_labels,
                                                        mining_size=self.mode.H)
                weight_total = weight_total * weight

            all_loss = F.binary_cross_entropy(probs, enc_target_labels, weight = weight_total.detach(), reduction='none')
            all_loss = torch.mean(torch.sum(all_loss, dim=1))
            
        else:
            all_loss = F.binary_cross_entropy(probs, enc_target_labels, reduction='none')
            all_loss = torch.mean(torch.sum(all_loss, dim=1))
        return all_loss

class ovr_loss_weight:

    def __init__(self, need_init=False, training_data=None, num_of_classes=10, is_verbose=False):
    
        self.num_of_classes = num_of_classes
        self.glob_pos_ratio = None
        self.glob_neg_ratio = None
        if need_init:
            assert training_data != None, f"Error : Initialization needs training data!"
            gt_labels = dataset.get_gt_labels(training_data, is_verbose=is_verbose)
            self.glob_pos_ratio, self.glob_neg_ratio = calc_class_ratio(num_of_classes, gt_labels, 1, is_verbose=is_verbose)

    def get_f_weight(self, probs, enc_targets, gamma=1):
        p_t = probs * enc_targets + (1 - probs) * (1 - enc_targets)
        weight = (1 - p_t) ** gamma
        return tools.device(weight)
    
    def get_c_weight(self, targets, enc_targets, from_global=True):

        if from_global:
            assert self.glob_pos_ratio != None or self.glob_neg_ratio != None, f"Error : <class OpenSetOvR_Weight> should be initialized. need_init=True"
            weight = torch.where(enc_targets==1, self.glob_pos_ratio, self.glob_neg_ratio)
        else:
            # Calculate class weighting in a batch-wise
            batch_pos_ratio, batch_neg_ratio = calc_class_ratio(self.num_of_classes, targets, 1, is_verbose=False)
            weight = torch.where(enc_targets==1, batch_pos_ratio, batch_neg_ratio)

        return tools.device(weight)
    
    def get_h_weight(self, probs, targets, enc_targets, mining_size=0.3):
        pos_cnts, neg_cnts = calc_class_cnt(self.num_of_classes, targets, is_verbose=False)
        weight = get_mining_mask(probs, enc_targets, pos_cnts, neg_cnts, mining_size)
        return tools.device(weight)

def calc_class_cnt(num_of_classes, labels, is_verbose=False):

    all_counts = len(labels) 
    pos_cnts = torch.Tensor([0]*num_of_classes)

    # get each known label counts = positives
    labels = labels[labels != -1]
    u_labels, cnts = labels.unique(return_counts=True)
    u_labels = u_labels.to(torch.int)

    for idx, l in enumerate(u_labels):
        pos_cnts[l] = cnts[idx]
    pos_cnts = pos_cnts.to(torch.int)
    
    if is_verbose:
        tools.print_table(u_labels.cpu().numpy(), pos_cnts.cpu().numpy())

    # get other label counts : negatives
    neg_cnts = all_counts - pos_cnts

    return tools.device(pos_cnts), tools.device(neg_cnts)

def calc_class_ratio(num_of_classes, labels, init_val=1, is_verbose=False):

    # initialize
    pos_ratio = torch.Tensor([init_val]*num_of_classes)
    neg_ratio = torch.Tensor([init_val]*num_of_classes)

    # get the source distribution
    pos_cnts, neg_cnts = calc_class_cnt(num_of_classes, labels, is_verbose)

    # get positive and negative ratios
    # if there is no positive samples for the class in the batch, 0 weighted for all samples in this class.
    for i, p_cnt in enumerate(pos_cnts):
        n_cnt = neg_cnts[i]
        if n_cnt > p_cnt:
            pos_ratio[i] = 1
            neg_ratio[i] = p_cnt/n_cnt
        else:
            pos_ratio[i] = n_cnt/p_cnt
            neg_ratio[i] = 1

    if is_verbose:
        print(f"Positive Ratio:\n{pos_ratio}")
        print(f"Negative Ratio:\n{neg_ratio}")
        print()

    return tools.device(pos_ratio), tools.device(neg_ratio)

def get_mining_mask(probs, enc_labels, pos_cnts, neg_cnts, alpha,):

    # Only consider negative's probability
    neg_probs = torch.where(enc_labels!=1, probs, -1)

    # Initialize the mask
    k_mask = torch.zeros(enc_labels.shape)
    k_mask = tools.device(k_mask)

    for i, p_cnt in enumerate(pos_cnts):
        n_cnt = neg_cnts[i]

        # Mining all negatives if there is no positives
        if p_cnt == 0:
            k_mask[:,i] == 1
            continue
        
        # Batch-wise class balancing weight if positives are more than negatives
        if p_cnt > n_cnt:
            k_mask[:,i] == p_cnt/n_cnt
            print(f"flag : positives are more than negatives {p_cnt} {n_cnt}")
            continue

        # Get k for the mining
        c_k = alpha*(n_cnt - p_cnt) + p_cnt 
        
        # Hard Negative Minings : Negatives with a high probability
        c_k_idxs = torch.topk(neg_probs[:,i], int(c_k)).indices
        k_mask[c_k_idxs, i] = p_cnt/c_k

    # Get the final masks including all postives and mined negatives
    mining_mask = enc_labels + k_mask
    return tools.device(mining_mask)

########################################################################
# Reference: 
# Vision And Security Technology (VAST) Lab in UCCS
# https://github.com/Vastlab/vast?tab=readme-ov-file
########################################################################

class entropic_openset_loss:

    def __init__(self, num_of_classes=10, unkn_weight=1):
        self.num_of_classes = num_of_classes
        self.eye = tools.device(torch.eye(self.num_of_classes))
        self.ones = tools.device(torch.ones(self.num_of_classes))
        self.unknowns_multiplier = unkn_weight / self.num_of_classes

    @tools.loss_reducer
    def __call__(self, logit_values, target, sample_weights=None):
        catagorical_targets = tools.device(torch.zeros(logit_values.shape))
        known_indexes = target != -1
        unknown_indexes = ~known_indexes
        catagorical_targets[known_indexes, :] = self.eye[target[known_indexes]]
        catagorical_targets[unknown_indexes, :] = (
            self.ones.expand((torch.sum(unknown_indexes).item(), self.num_of_classes))
            * self.unknowns_multiplier
        )
        log_values = F.log_softmax(logit_values, dim=1)
        negative_log_values = -1 * log_values
        loss = negative_log_values * catagorical_targets
        sample_loss = torch.sum(loss, dim=1)
        if sample_weights is not None:
            sample_loss = sample_loss * sample_weights
        return sample_loss.mean()

class objectoSphere_loss:
    def __init__(self, knownsMinimumMag=50.0):
        self.knownsMinimumMag = knownsMinimumMag

    @tools.loss_reducer
    def __call__(self, features, target, sample_weights=None):
        # compute feature magnitude
        mag = features.norm(p=2, dim=1)
        # For knowns we want a certain magnitude
        mag_diff_from_ring = torch.clamp(self.knownsMinimumMag - mag, min=0.0)

        # Loss per sample
        loss = tools.device(torch.zeros(features.shape[0]))
        known_indexes = target != -1
        unknown_indexes = ~known_indexes
        # knowns: punish if magnitude is inside of ring
        loss[known_indexes] = mag_diff_from_ring[known_indexes]
        # unknowns: punish any magnitude
        loss[unknown_indexes] = mag[unknown_indexes]
        loss = torch.pow(loss, 2)
        if sample_weights is not None:
            loss = sample_weights * loss
        return loss.mean()
