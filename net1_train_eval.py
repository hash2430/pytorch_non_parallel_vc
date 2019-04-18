import time
import os
import copy
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
from hparams import hparam as hp
from dataset import Net1TimitData
import logging
from tensorboardX import SummaryWriter

logger = logging.getLogger('train1')

# TODO: Model Save must be called from here
def model_train(checkpoint_path, net1_model, train_loader, valid_loader,
                criterion, starting_epoch=0, device=0, lr=0.001, loss=100):
    logdir = checkpoint_path.rstrip('/checkpoint.tar')
    writer = SummaryWriter(logdir)

    num_epochs = hp.train1.num_epochs
    if starting_epoch >= num_epochs:
        return net1_model, starting_epoch, loss
    eval_interval = hp.train1.eval_interval

    optimizer = torch.optim.Adam(net1_model.parameters(), lr=lr)
    plateu_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)

    '''
    Do train for num_epochs and call eval for each 'eval_interval'th epoch
    Save the best model so far according to the evaluation result
    '''
    best_model_wts = copy.deepcopy(net1_model.state_dict())
    best_loss = loss

    # Run validation for every "eval_interval" times of training
    for epoch in range(starting_epoch + 1, num_epochs + 1):
        logger.info('-' * 10)
        logger.info('Epoch {}/{}'.format(epoch, num_epochs))

        # Training
        net1_model.train() # Set to train mode
        start = time.time()
        # Iterate over dataset
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            mfccs = data['mfccs']
            mfccs = Variable(mfccs).type(torch.FloatTensor)

            y_phn = data['phns']
            y_phn = Variable(y_phn).type(torch.FloatTensor)
            dtype = torch.LongTensor
            if device > 0:
                mfccs = mfccs.cuda(device - 1)
                y_phn = y_phn.cuda(device - 1)
                net1_model = net1_model.cuda(device - 1)
                dtype = torch.cuda.LongTensor

            ppgs, preds, logits = net1_model(mfccs)
            logits = logits.view(-1, Net1TimitData.phone_vocab_size)
            y_phn = y_phn.view(-1).type(dtype) # (N,T)=>(NxT)
            # TODO: Create isTarget mask before calculating loss and accuracy
            loss = criterion(mfccs, logits, y_phn)
            loss.backward()

            if hp.train1.do_gradient_clip:
                torch.nn.utils.clip_grad_norm(net1_model.parameters(), hp.train1.clip_norm)
                lr = optimizer.param_groups[0]['lr']
                for p in net1_model.parameters():
                    p.data.add_(-lr, p.grad.data)
            else:
                optimizer.step()

        end = time.time()
        diff = end - start
        logger.info('Training Time : {} s'.format(diff))
        logger.info('Training Loss: {} '.format(loss.data[0]))
        writer.add_scalar('Training_Loss', loss.data[0], epoch)

        # Validation
        if (epoch % eval_interval == 0) or (epoch == num_epochs):
            eval_loss, acc = model_eval(net1_model, valid_loader, criterion, device=device)
            writer.add_scalar('Net1_Eval_Loss', eval_loss, epoch)
            writer.add_scalar('Net1_Accuracy', acc, epoch)
            logger.info("Evaluation acc: {}".format(acc))
            logger.info('Evaluation loss: {}'.format(eval_loss))
            plateu_lr_scheduler.step(eval_loss)
            curr_lr = optimizer.param_groups[0]['lr']
            logger.info('Learning rate: {}'.format(curr_lr))
            if curr_lr < hp.train1.stopping_lr:
                logger.info("Early stopping\n\n")
                break
            if eval_loss < best_loss:
                best_loss = eval_loss
                best_model_wts = copy.deepcopy(net1_model.state_dict())
                net1_model.load_state_dict(best_model_wts)
                model_dict = {
                    'epoch': epoch,
                    'model_state_dict': net1_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss
                }
                # extract directory part and create
                if not os.path.exists(checkpoint_path):
                    dirname = os.path.dirname(checkpoint_path)
                    os.makedirs(dirname, exist_ok=True)

                torch.save(model_dict, checkpoint_path)

    net1_model.load_state_dict(best_model_wts)
    writer.close()
    return net1_model, epoch, best_loss

def model_eval(net1_model, valid_loader, criterion, device):
    eval_loss = 0.0
    acc = 0.0

    for i, data in enumerate(valid_loader):
        net1_model.eval()
        mfccs = data['mfccs']
        mfccs = Variable(mfccs).type(torch.FloatTensor)
        y_phns = data['phns']
        if device > 0:
            y_phns_tensor = y_phns.type(torch.cuda.LongTensor).view(-1)
        else:
            y_phns_tensor = y_phns.type(torch.LongTensor).view(-1)
        y_phns = Variable(y_phns).type(torch.FloatTensor)
        dtype = torch.LongTensor
        if device > 0:
            mfccs = mfccs.cuda(device - 1)
            y_phns = y_phns.cuda(device - 1)
            net1_model = net1_model.cuda(device - 1)
            dtype = torch.cuda.LongTensor

        ppgs, pred_phns, logits = net1_model(mfccs)
        logits = logits.view(-1, Net1TimitData.phone_vocab_size)
        y_phns = y_phns.view(-1).type(dtype)
        loss = criterion(mfccs, logits, y_phns)
        eval_loss += loss.data[0]

        # Calculate accuracy for logging purpose
        is_target = torch.sign(torch.abs(torch.sum(mfccs, dim=-1))).view(-1).data.byte()
        num_targets = torch.sum(is_target)

        pred_phns = pred_phns.view(-1).type(dtype)


        _acc = torch.mul(torch.eq(pred_phns, y_phns_tensor), is_target)
        _acc = _acc.sum()
        acc += _acc/num_targets

    avg_loss = eval_loss / len(valid_loader)
    acc = acc / len(valid_loader)
    return avg_loss, acc

def where(cond, x_1, x_2):
    return (cond * x_1) + ((1-cond) * x_2)
