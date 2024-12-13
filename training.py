import time
import pathlib
import numpy as np

import torch
from torch.nn import functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from library import architectures, tools, losses

def train(args, net, optimizer, train_data_loader, loss_func, num_classes, debug=False):

    net.train()

    loss_history = []
    train_accuracy = torch.zeros(2, dtype=int)
    train_confidence = torch.zeros(4, dtype=float)
    debug_results = None
    grad_results, fc2_weights_results, layers = [], [], []

    for x, y in train_data_loader:
        x = tools.device(x)
        y = tools.device(y)
        optimizer.zero_grad()
        logits, _ = net(x)
        loss = loss_func(logits, y)

        if config.approach == "OvR":
            scores = F.sigmoid(logits)
            train_confidence += losses.confidence(scores, y,
                                                    offset = 0.,
                                                    unknown_class = -1,
                                                    last_valid_class = None,)
        else:
            scores = torch.nn.functional.softmax(logits, dim=1)
            train_confidence += losses.confidence(scores, y,
                                                    offset = 1. / num_classes,
                                                    unknown_class = -1,
                                                    last_valid_class = None,)
            
        train_accuracy += losses.accuracy(scores, y)

        loss_history.append(loss)
        loss.backward()

        # DEBUG : Check for vanishing or exploding gradient
        if debug:
            batch_grad, layers = tools.check_grad_flow(net.named_parameters())
            batch_fc2_weights = tools.check_fc2_weights(net)   
            grad_results.append(batch_grad)
            fc2_weights_results.append(batch_fc2_weights)
            debug_results = (grad_results, layers, fc2_weights_results)
        
        optimizer.step()

    return loss_history, train_accuracy, train_confidence, debug_results

def validate(args, net, val_data_loader, loss_func, epoch, num_classes):

    net.eval()
    with torch.no_grad():
        
        val_loss = torch.zeros(2, dtype=float)
        val_accuracy = torch.zeros(2, dtype=int)
        val_confidence = torch.zeros(4, dtype=float)

        for x,y in val_data_loader:
            # predict
            x = tools.device(x)
            y = tools.device(y)
            logits, _ = net(x)
            loss = loss_func(logits, y)

            # metrics on validation set
            if ~torch.isnan(loss):
                val_loss += torch.tensor((loss * len(y), len(y)))
                
            if config.approach == "OvR":
                scores = F.sigmoid(logits)
                val_confidence += losses.confidence(scores, y,
                                                        offset = 0.,
                                                        unknown_class = -1,
                                                        last_valid_class = None,)
            else:
                scores = torch.nn.functional.softmax(logits, dim=1)
                val_confidence += losses.confidence(scores, y,
                                                        offset = 1. / num_classes,
                                                        unknown_class = -1,
                                                        last_valid_class = None,)
            val_accuracy += losses.accuracy(scores, y)
    
    return val_loss, val_accuracy, val_confidence

def worker(args, config, seed):

    BEST_SCORE = 0
    tools.set_seeds(seed)

    # ===================================================
    # WORKING PARAMETERS
    # ===================================================
    # Environment
    num_workers = config.num_workers
    lr = config.opt.lr

    # Dataset
    if config.scale == 'smallscale':
        batch_size = config.batch_size.smallscale
        epochs = config.epochs.smallscale
    else:
        batch_size = config.batch_size.largescale
        epochs = config.epochs.largescale
    # Architecture
    if 'LeNet' in config.name:
        arch_name = 'LeNet'
        if 'plus_plus' in config.name:
            arch_name = 'LeNet_plus_plus'
    elif 'ResNet_50' in config.name:
        arch_name = 'ResNet_50'
    else:
        arch_name = None

    # ===================================================
    # SETTINGs
    # ===================================================
    # 1. Model saving directory
    results_dir = pathlib.Path(f"{config.scale}/_s{seed}/{config.name}/{config.approach}")
    model_file = f"{results_dir}/{config.approach}.model"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = results_dir/'Logs'
    writer = SummaryWriter(logs_dir)

    # 2. Dataset and Loss function
    loss_func, training_data, validation_data, num_classes = list(zip(*tools.get_data_and_loss(config, seed).items()))[-1]
    train_data_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_data_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # 3. Neural Network
    net = architectures.__dict__[arch_name](num_classes=num_classes,
                                            final_layer_bias=False,
                                            feat_dim=config.arch.feat_dim)
    net = tools.device(net)

    # 4. Optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # ?. (Optional) Debug
    if config.training_debug_1:
        model_file_history = model_file.replace('.model', f"_0.model")
        torch.save(net.state_dict(), model_file_history)

    # Setting Print out
    
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(
        f"Configuration Details \n"
        f"Train: {len(training_data)}\tVal: {len(validation_data)}\n"
        f"Batch Size: {batch_size} \n"
        f"Epochs: {epochs} \n"
        f"Adam Learning Rate: {lr} \n"
        f"Results: {model_file} \n"
        f"Debug? {config.training_debug_1}, {config.training_debug_2}"
          )


    # ===================================================
    # TRAINING
    # ===================================================
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Trainig Start!")
    for epoch in range(1, epochs + 1, 1):
        # Check a duration of one epoch.
        t0 = time.time() 

        # 1.Batch training
        loss_history, train_accuracy, train_confidence, debug_info = train(args, net, 
                                                                  optimizer, 
                                                                  train_data_loader, 
                                                                  loss_func, num_classes, debug=config.training_debug_2)

        # 2. Validate the model
        val_loss, val_accuracy, val_confidence = validate(args, net, 
                                                             val_data_loader, 
                                                             loss_func, 
                                                             epoch, num_classes)

        # 3. Model score calcuation
        save_status = "NO"
        curr_score = float(val_confidence[0] / val_confidence[1]) 
        if config.data.train_neg_size != 0: # Consider unknown confidence together
            curr_score += float(val_confidence[2] / val_confidence[3])
        
        # 4. Save/update the model
        if curr_score > BEST_SCORE:
            torch.save(net.state_dict(), model_file)
            BEST_SCORE = curr_score
            save_status = "YES"

            # ?. (Optional) Debug
            if config.training_debug_1:
                model_file_history = model_file.replace('.model', f"_{epoch}.model")
                torch.save(net.state_dict(), model_file_history)

        # ?. (Optional) Debug
        if config.training_debug_2:
            debug_results_dir = results_dir/'Debug'
            debug_results_dir.mkdir(parents=True, exist_ok=True)
            np.save(debug_results_dir/f'grad_results_{epoch}.npy',np.array(debug_info[0]))
            np.save(debug_results_dir/f'layers.npy',np.array(debug_info[1]))
            np.save(debug_results_dir/f'fc2_weights_results_{epoch}.npy',np.array(debug_info[2]))
            print("Training debug info save done!")

        # 5. Log tensorboard form
        epoch_running_loss = torch.mean(torch.tensor(loss_history))
        writer.add_scalar('Loss/train', epoch_running_loss, epoch)
        writer.add_scalar('Loss/val', val_loss[0] / val_loss[1], epoch)
        writer.add_scalar('Acc/train', float(train_accuracy[0] / train_accuracy[1]), epoch)
        writer.add_scalar('Acc/val', float(val_accuracy[0] / val_accuracy[1]), epoch)
        writer.add_scalar('Conf/train_kn', float(train_confidence[0] / train_confidence[1]), epoch)
        writer.add_scalar('Conf/train_neg', float(train_confidence[2] / train_confidence[3]), epoch)
        writer.add_scalar('Conf/val_kn', float(val_confidence[0] / val_confidence[1]), epoch)
        writer.add_scalar('Conf/val_neg', float(val_confidence[2] / val_confidence[3]), epoch)
        
        # 7. Print out training progress
        print(f"Epoch {epoch} ({time.time()-t0:.2f}sec): "
              f"TRAIN SET -- "
              f"Loss {epoch_running_loss:.5f} "
              f"Acc {float(train_accuracy[0] / train_accuracy[1]):.5f} "
              f"KnConf {float(train_confidence[0] / train_confidence[1]):.5f} "
              f"UnConf {float(train_confidence[2] / train_confidence[3]):.5f} "
              f"VALIDATION SET -- "
              f"Loss {float(val_loss[0] / val_loss[1]):.5f} "
              f"Acc {float(val_accuracy[0] / val_accuracy[1]):.5f} "
              f"KnConf {float(val_confidence[0] / val_confidence[1]):.5f} "
              f"UnConf {float(val_confidence[2] / val_confidence[3]):.5f} "
              f"SAVING MODEL -- {curr_score:.3f} {save_status}")


if __name__ == "__main__":

    args = tools.train_command_line_options()
    config = tools.load_yaml(args.config)

    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(
        f"Execution Time: {time.strftime('%d %b %Y %H:%M:%S')} \n"
        f"GPU: {args.gpu} \n"
        f"Dataset Scale: {config.scale} \n"
        f"Architecture: {config.name} \n"
        f"Approach: {config.approach} \n"
        f"Configuration: {args.config} \n"
        f"Seed: {args.seed}\n"
          )

    try: 
        change_lr = 'largescale' in config.scale  and config.loss.ovr.mode.H != None
    except: 
        change_lr = False

    if change_lr:
        config.opt.lr = 1.e-4
        print(f"Learning rate changed : (new) {config.opt.lr}")
        
    if args.gpu is not None and torch.cuda.is_available():
        tools.set_device_gpu(args.gpu)
    else:
        print("Running in CPU mode, training might be slow")
        tools.set_device_cpu()

    for s in args.seed:

        worker(args, config, s)
        print("Training Done!\n\n\n")

    print("All training done!")