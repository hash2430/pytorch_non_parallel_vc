import copy
import logging
import os
import time

import librosa
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim import lr_scheduler

from audio_utils import *
from utils import Profiler, prepare_spec_image

'''
TODO: decide whether to retrieve net1_model from outside of the net2_model or retrieve net1 from the submodule of net2
'''
def model_train(checkpoint_path, net1_model, net2_model, train_loader, valid_loader, criterion,
                starting_epoch=0, device=0, lr=0.001, loss=100):
    logger = logging.getLogger('train2')
    logdir = checkpoint_path.rstrip('/checkpoint.tar')
    writer = SummaryWriter(logdir)

    num_epochs = hp.train2.num_epochs
    if starting_epoch >= num_epochs:
        return net2_model, starting_epoch, loss
    eval_interval = hp.train2.eval_interval

    optimizer= torch.optim.Adam(net2_model.parameters(), lr=lr)
    plateau_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)

    '''
    Do train for num_epochs and call eval for each 'eval_interval'th epoch
    Save the best model so far according to the evaluation result
    '''
    best_model_wts = copy.deepcopy(net2_model.state_dict())
    best_loss = loss

    # Run validation for every "eval_interval" times of training
    for epoch in range(starting_epoch + 1, num_epochs + 1):
        logger.info('-' * 10)
        logger.info('Epoch {}/{}'.format(epoch, num_epochs))
        # training
        net1_model.eval()
        net2_model.train()
        start = time.time()
        # Iterate over dataset
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            mfccs = data['mfccs']
            y_spec = data['y_spec']
            y_mel = data['y_mel']
            mfccs = Variable(mfccs).type(torch.FloatTensor)
            y_spec = Variable(y_spec).type(torch.FloatTensor)
            y_mel = Variable(y_mel).type(torch.FloatTensor)
            dtype = torch.FloatTensor
            if device > 0:
                mfccs = mfccs.cuda(device - 1)
                y_spec = y_spec.cuda(device - 1)
                y_mel = y_mel.cuda(device - 1)
                net1_model = net1_model.cuda(device - 1)
                net2_model = net2_model.cuda(device - 1)
                dtype = torch.cuda.FloatTensor
            ppgs, preds, _ = net1_model(mfccs)
            ppgs = ppgs.type(dtype)
            ppgs = ppgs.detach()
            pred_spec, pred_mel = net2_model(ppgs)
            loss = criterion(pred_spec, pred_mel, y_spec, y_mel)
#            loss.requires_grad=True
            loss.backward()

            # 'clip_grad_norm' to prevent exploding gradient problem in GRU
            # This part replaces 'optimizer.step()'
            if hp.train2.do_gradient_clip:
                torch.nn.utils.clip_grad_norm(net2_model.parameters(), hp.train2.clip_norm)
                lr = optimizer.param_groups[0]['lr']
                for p in net2_model.parameters():
                    p.data.add_(-lr, p.grad.data)
            else:
                optimizer.step()

        end = time.time()
        diff = end - start
        logger.info('Training Time : {} s'.format(diff))
        logger.info('Training Loss: {}'.format(loss.item()))
        writer.add_scalar('Training_Loss', loss.item(), epoch)
        # validation
        if (epoch % eval_interval == 0) or (epoch == num_epochs):
            eval_loss = model_eval(net1_model, net2_model, valid_loader, criterion, device=device)
            writer.add_scalar('Net2_Eval_Loss', eval_loss, epoch)
            plateau_lr_scheduler.step(eval_loss)
            curr_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Learning_Rate', curr_lr, epoch)
            logger.info('Evaluation loss: {:.4f} \n'.format(eval_loss))
            logger.info('Learning rate: {}'.format(curr_lr))
            if curr_lr < hp.train2.stopping_lr:
                logger.info("Early stopping\n\n")
                break
            if eval_loss < best_loss:
                best_loss = eval_loss
                best_model_wts = copy.deepcopy(net2_model.state_dict())

                net2_model.load_state_dict(best_model_wts)
                model_dict = {
                    'epoch': epoch,
                    'model_state_dict': net2_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss
                }
                # extract directory part and create
                if not os.path.exists(checkpoint_path):
                    dirname = os.path.dirname(checkpoint_path)
                    os.makedirs(dirname, exist_ok=True)

                torch.save(model_dict, checkpoint_path)

    net2_model.load_state_dict(best_model_wts)
    writer.close()
    return net2_model, epoch, best_loss

# Compare the net2_model output and the ground truth y_spec from valid_loader
def model_eval(net1_model, net2_model, valid_loader, criterion, device=0):
    eval_loss = 0.0

    for i, data in enumerate(valid_loader):
        net1_model.eval()
        net2_model.eval()
        mfccs = data['mfccs']
        y_spec = data['y_spec']
        y_mel = data['y_mel']

        mfccs = Variable(mfccs).type(torch.FloatTensor)
        y_spec = Variable(y_spec).type(torch.FloatTensor)
        y_mel = Variable(y_mel).type(torch.FloatTensor)
        if device > 0:
            mfccs = mfccs.cuda(device - 1)
            y_spec = y_spec.cuda(device - 1)
            y_mel = y_mel.cuda(device - 1)
            net1_model = net1_model.cuda(device - 1)
            net2_model = net2_model.cuda(device - 1)

        ppgs, _, _ = net1_model(mfccs)
        pred_spec, pred_mel = net2_model(ppgs)

        loss = criterion(pred_spec, pred_mel, y_spec, y_mel)
        eval_loss += loss.item()

    avg_loss = eval_loss / len(valid_loader)
    return avg_loss

def convert(net1_model, net2_model, conversion_source_loader, logdir_train2, logdir_convert):
    writer = SummaryWriter(logdir_train2)
    logger = logging.getLogger('CONVERT')
    # Dataloader => source mfcc
    net1_profiler = Profiler('Network 1')
    net2_profiler = Profiler('Network 2')
    spec_postprocessing_profiler = Profiler('Spectrogram postprocessing')
    vocoder_profiler = Profiler('Vocoder')
    for i, data in enumerate(conversion_source_loader):
        net1_model.eval()
        net2_model.eval()
        mfccs = data['mfccs']
        mfccs = Variable(mfccs).type(torch.FloatTensor)

        net1_profiler.start()
        ppgs, _, _ = net1_model(mfccs)
        net1_profiler.end()

        net2_profiler.start()
        pred_spec, pred_mel = net2_model(ppgs) # (N, T, n_fft//2 + 1)
        net2_profiler.end()

        spec_postprocessing_profiler.start()
        pred_spec = pred_spec.data.cpu().numpy()
        pred_spec_img = prepare_spec_image(pred_spec[0])
        writer.add_image('0_Pred_spec', pred_spec_img)
        pred_mel = pred_mel.data.cpu().numpy()
        pred_mel_img = prepare_spec_image(pred_mel[0])
        writer.add_image('0_Pred_mel', pred_mel_img)

        # 1. Denormalization
        pred_spec = denormalize_db(pred_spec, hp.default.max_db, hp.default.min_db)
        pred_spec_img = prepare_spec_image(pred_spec[0])
        writer.add_image('1_Denormalization', pred_spec_img)

        # 2. Db to amp
        pred_spec = db2amp(pred_spec)
        pred_spec_img = prepare_spec_image(pred_spec[0])
        writer.add_image('2_Db_to_amp', pred_spec_img)

        # 3. Emphasize the magnitude
        pred_spec = np.power(pred_spec, hp.convert.emphasis_magnitude)
        pred_spec_img = prepare_spec_image(pred_spec[0])
        writer.add_image('3_Emphasize_magnitude', pred_spec_img)
        spec_postprocessing_profiler.end()

        vocoder_profiler.start()
        for j in range(pred_spec.shape[0]):
            # Spectrogram to waveform
            audio = spec2wav(pred_spec[j].T,
                             hp.default.n_fft,
                             hp.default.win_length,
                             hp.default.hop_length,
                             hp.default.n_iter)
            # Save the result
            idx = pred_spec.shape[0] * (i) + j
            # TODO: Apply inverse pre-emphasis
            writer.add_audio('Predicted_audio_Before_inv_preemphasis{}'.format(idx), audio)
            audio = inv_preemphasis(audio, coeff=hp.default.preemphasis)
            audio = audio.astype('float32')
            writer.add_audio('Predicted_audio_After_inv_preemphasis{}'.format(idx), audio)

            path = '{}/m2f_{:03d}.wav'.format(logdir_convert, idx)
            #soundfile.write(path, wav, hp.default.sr, format='wav')
            librosa.output.write_wav(path, audio, sr=hp.default.sr)
        vocoder_profiler.end()

    logger.debug('Net1: {} s'.format(net1_profiler.get_time()))
    logger.debug('Net2: {} s'.format(net2_profiler.get_time()))
    logger.debug('Spectrogram postprocessing: {} s'.format(spec_postprocessing_profiler.get_time()))
    logger.debug('Vocoder: {} s'.format(vocoder_profiler.get_time()))
    writer.close()
    return