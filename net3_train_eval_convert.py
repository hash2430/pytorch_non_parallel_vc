import copy
import logging
import time

from torch.autograd import Variable
from torch.optim import lr_scheduler

logger = logging.getLogger('train3')
from audio_utils import *
from tensorboardX import SummaryWriter
from utils import Profiler, prepare_spec_image

# train_eval corresponds to ModelDesc._build_graph() of tensorpack
def model_train(checkpoint_path, net1_model, net2_model, net3_model,
                train_loader, valid_loader, criterion,
                starting_epoch=0,device=0,lr=0.001, loss=100):
    logdir = checkpoint_path.rstrip('/checkpoint.tar')
    writer = SummaryWriter(logdir)

    num_epochs = hp.train3.num_epochs
    if starting_epoch >= num_epochs:
        return net3_model, starting_epoch, loss
    eval_interval = hp.train3.eval_interval
    # Not sure if I have to make a separate submodule outof true net3 and give it only
    optimizer = torch.optim.Adam(net3_model.parameters(), lr=lr)
    plateu_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)

    '''
    Do train for num_epochs and call eval for each 'eval_interval'th epoch
    Save the best model so far according to the evaluation result
    '''
    best_model_wts = copy.deepcopy(net3_model.state_dict())
    best_loss = loss

    # run validation for every "eval_interval" times of training
    for epoch in range(starting_epoch + 1, num_epochs + 1):
        logger.info('-' * 10)
        logger.info('Epoch {}/{}'.format(epoch, num_epochs))
        # training
        #plateu_lr_scheduler.step()
        net1_model.eval()
        net2_model.eval()
        net3_model.train() # Set model to training mode
        start = time.time()

        # Iterate over dataset
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            mfccs = data['mfccs']
            mfccs = Variable(mfccs).type(torch.FloatTensor)
            dtype = torch.FloatTensor
            if device > 0:
                mfccs = mfccs.cuda(device - 1)
                net1_model = net1_model.cuda(device - 1)
                net2_model = net2_model.cuda(device - 1)
                net3_model = net3_model.cuda(device - 1)


            ppgs, _, _ = net1_model(mfccs)
            y_spec, y_mel = net2_model(ppgs)
            y_spec = y_spec.detach()
            y_mel = y_mel.detach()
            pred_spec, pred_mel = net3_model(mfccs)

            loss = criterion(pred_spec, pred_mel, y_spec, y_mel)
            loss.backward()
            if hp.train3.do_gradient_clip:
                torch.nn.utils.clip_grad_norm(net3_model.parameters(), hp.train3.clip_norm)
                lr = optimizer.param_groups[0]['lr']
                for p in net3_model.parameters():
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
            eval_loss = model_eval(net1_model, net2_model, net3_model, valid_loader, criterion, device)
            writer.add_scalar('Net3_Eval_Loss', eval_loss, epoch)
            plateu_lr_scheduler.step(eval_loss)
            curr_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Learning_Rate', curr_lr, epoch)
            logger.info('Evaluation loss: {:.4f} \n'.format(eval_loss))
            logger.info('Learning rate: {}'.format(curr_lr))
            if curr_lr < hp.train3.stopping_lr:
                logger.info("Early stopping\n\n")
                break
            if eval_loss < best_loss:
                best_loss = eval_loss
                best_model_wts = copy.deepcopy(net3_model.state_dict())

                net3_model.load_state_dict(best_model_wts)
                model_dict = {
                    'epoch': epoch,
                    'model_state_dict': net3_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss
                }
                # extract directory part and create
                if not os.path.exists(checkpoint_path):
                    dirname = os.path.dirname(checkpoint_path)
                    os.makedirs(dirname, exist_ok=True)

                torch.save(model_dict, checkpoint_path)

    net3_model.load_state_dict(best_model_wts)
    return net3_model, epoch, best_loss

# loss: difference between Net2 generated pdf
def model_eval(net1_model, net2_model, net3_model, valid_loader, criterion, device=0):
    eval_loss = 0.0

    for i, data in enumerate(valid_loader):
        net1_model.eval()
        net2_model.eval()
        net3_model.eval()
        mfccs = data['mfccs']
        mfccs = Variable(mfccs).type(torch.FloatTensor)

        if device > 0:
            mfccs = mfccs.cuda(device - 1)

        ppgs, _, _ = net1_model(mfccs)
        y_spec, y_mel = net2_model(ppgs)
        y_spec = y_spec.detach()
        y_mel = y_mel.detach()
        pred_spec, pred_mel = net3_model(mfccs)

        loss = criterion(pred_spec, pred_mel, y_spec, y_mel)
        eval_loss += loss.item()

    avg_loss = eval_loss / len(valid_loader)
    return avg_loss

def quick_convert(net3_model, conversion_source_loader, logdir):
    writer = SummaryWriter(logdir)
    logger = logging.getLogger('QUICK_CONVERT')
    net3_profiler = Profiler('Network 3')
    spec_postprocessing_profiler = Profiler('Spectrogram postprocessing')
    vocoder_profiler = Profiler('Vocoder')
    # Dataloader => source mfcc
    for i, data in enumerate(conversion_source_loader):
        net3_model.eval()
        mfccs = data['mfccs']
        mfccs = Variable(mfccs).type(torch.FloatTensor)
        net3_profiler.start()
        pred_spec, pred_mel = net3_model(mfccs)
        net3_profiler.end()

        spec_postprocessing_profiler.start()
        pred_spec = pred_spec.data.cpu().numpy()
        # Denormalization
        pred_spec = denormalize_db(pred_spec, hp.default.max_db, hp.default.min_db)

        # Db to amp
        pred_spec = db2amp(pred_spec)

        # Emphasize the magnitude
        pred_spec = np.power(pred_spec, hp.convert.emphasis_magnitude)
        spec_postprocessing_profiler.end()

        vocoder_profiler.start()
        for j in range(pred_spec.shape[0]):
            # Spectrogram to waveform
            audio = spec2wav(pred_spec[j].T,
                             hp.default.n_fft,
                             hp.default.win_length,
                             hp.default.hop_length,
                             hp.default.n_iter)
            # TODO: Apply inverse pre-emphasis
            audio = inv_preemphasis(audio, coeff=hp.default.preemphasis)
            audio = audio.astype('float32')

            path = '{}/m2f_{:03d}.wav'.format(logdir, pred_spec.shape[0] * (i) + j)
            # soundfile.write(path, wav, hp.default.sr, format='wav')
            librosa.output.write_wav(path, audio, sr=hp.default.sr)
        vocoder_profiler.end()

    logger.debug('Net3: {} s'.format(net3_profiler.get_time()))
    return


