import argparse
import collections
import datetime
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from models.VGG import VGG
import shutil
import time
import dataset
import mlconfig
import toolbox
import torch
import torchvision
from torchvision.utils import save_image
import util
from util import club
import madrys
import numpy as np
from evaluator import Evaluator
from tqdm import tqdm
from trainer import Trainer
from models.starGAN import DomainGenerator
from utils.wassersteinLoss import *
import clip
import cv2
from models.vision_transformer import *
mlconfig.register(madrys.MadrysLoss)


# General Options
parser = argparse.ArgumentParser(description='ClasswiseNoise')
parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--version', type=str, default="resnet18")
parser.add_argument('--exp_name', type=str, default="experiment/")
parser.add_argument('--config_path', type=str, default='configs/cifar10')
parser.add_argument('--load_model', action='store_true', default=False)
parser.add_argument('--data_parallel', action='store_true', default=False)
parser.add_argument('--epoch', default=30, type=int)
# Datasets Options
parser.add_argument('--train_batch_size', default=512, type=int, help='perturb step size')
parser.add_argument('--eval_batch_size', default=512, type=int, help='perturb step size')
parser.add_argument('--num_of_workers', default=6, type=int, help='workers for loader')
parser.add_argument('--train_data_type', type=str, default='CIFAR10')
parser.add_argument('--train_data_path', type=str, default='/data/datasets')
parser.add_argument('--test_data_type', type=str, default='CIFAR10')
parser.add_argument('--test_data_path', type=str, default='/data/datasets')
# Perturbation Options
parser.add_argument('--universal_train_portion', default=0.5, type=float, help='only valid when args.use_subset is True')
parser.add_argument('--universal_stop_error', default=0.1, type=float)
parser.add_argument('--universal_train_target', default=None, type=str, choices=['train_subset', 'train_dataset'], help='use subset or whole training set when training noise generator')
parser.add_argument('--train_step', default=10, type=int)
parser.add_argument('--use_subset', action='store_true', default=False)
parser.add_argument('--attack_type', default='min-min', type=str, choices=['min-min', 'min-max', 'random'], help='Attack type')
parser.add_argument('--perturb_type', default='classwise', type=str, choices=['classwise', 'samplewise'], help='Perturb type')
parser.add_argument('--patch_location', default='center', type=str, choices=['center', 'random'], help='Location of the noise')
parser.add_argument('--noise_shape', default=[10, 3, 32, 32], nargs='+', type=int, help='noise shape')
parser.add_argument('--epsilon', default=8, type=float, help='perturbation')
parser.add_argument('--num_steps', default=1, type=int, help='perturb number of steps')
parser.add_argument('--step_size', default=0.8, type=float, help='perturb step size')
parser.add_argument('--random_start', action='store_true', default=False)
parser.add_argument('--image_size', default=32, type=int, help='image shape')

args = parser.parse_args()
if args.use_subset:
    args.universal_train_target = 'train_subset'
else:
    args.universal_train_target = 'train_dataset'

# Convert Eps
args.epsilon = args.epsilon / 255     
args.step_size = args.step_size / 255  

# Set up Experiments
args.exp_name = args.exp_name + args.perturb_type + '/noise_generation' + str(datetime.datetime.now())

exp_path = args.exp_name
log_file_path = os.path.join(exp_path, args.version)
checkpoint_path = os.path.join(exp_path, 'checkpoints')
checkpoint_path_file = os.path.join(checkpoint_path, args.version)
generator_path_file = os.path.join(args.exp_name, 'generator')
util.build_dirs(exp_path)
# util.build_dirs(checkpoint_path)
logger = util.setup_logger(name=args.version, log_file=log_file_path + ".log")

# CUDA Options
logger.info("PyTorch Version: %s" % (torch.__version__))
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
    logger.info("GPU List: %s" % (device_list))
else:
    device = torch.device('cpu')

# Load Exp Configs
config_file = os.path.join(args.config_path, args.version)+'.yaml'
config = mlconfig.load(config_file)
# config.set_immutable()
for key in config:
    logger.info("%s: %s" % (key, config[key]))
shutil.copyfile(config_file, os.path.join(exp_path, args.version  +'.yaml'))

def train(starting_epoch, model, optimizer, scheduler, criterion, trainer, evaluator, ENV, data_loader):
    for epoch in range(starting_epoch, config.epochs):
        logger.info("")
        logger.info("="*20 + "Training Epoch %d" % (epoch) + "="*20)

        # Train
        ENV['global_step'] = trainer.train(epoch, model, criterion, optimizer)
        ENV['train_history'].append(trainer.acc_meters.avg*100)
        scheduler.step()

        # Eval
        logger.info("="*20 + "Eval Epoch %d" % (epoch) + "="*20)
        evaluator.eval(epoch, model)
        payload = ('Eval Loss:%.4f\tEval acc: %.2f' % (evaluator.loss_meters.avg, evaluator.acc_meters.avg*100))
        logger.info(payload)
        ENV['eval_history'].append(evaluator.acc_meters.avg*100)
        ENV['curren_acc'] = evaluator.acc_meters.avg*100

        # Reset Stats
        trainer._reset_stats()
        evaluator._reset_stats()

        # Save Model
        target_model = model.module if args.data_parallel else model
        util.save_model(ENV=ENV,
                        epoch=epoch,
                        model=target_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        filename=checkpoint_path_file)
        logger.info('Model Saved at %s', checkpoint_path_file)
    return


def universal_perturbation_eval(noise_generator, random_noise, data_loader, model, eval_target=args.universal_train_target):
    loss_meter = util.AverageMeter()
    err_meter = util.AverageMeter()
    random_noise = random_noise.to(device)
    model = model.to(device)   
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader[eval_target]):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if random_noise is not None:
                for i in range(len(labels)):
                    class_index = labels[i].item()
                    noise = random_noise[class_index]
                    mask_cord, class_noise = noise_generator._patch_noise_extend_to_img(noise, image_size=images[i].shape, patch_location=args.patch_location)
                    images[i] += class_noise
            _, pred = model(images)
            err = (pred.data.max(1)[1] != labels.data).float().sum()
            loss = torch.nn.CrossEntropyLoss()(pred, labels)
            loss_meter.update(loss.item(), len(labels))
            err_meter.update(err / len(labels))
    return loss_meter.avg, err_meter.avg


def Loss_distribution(Classifier, x_ori,x_gen):
    f_ori,_ = Classifier(x_ori)
    f_gen,_ = Classifier(x_gen)
    C = cost_matrix(f_ori, f_gen).cuda()
    loss = sink(C)
    return loss

def universal_perturbation(args, noise_generator, trainer, evaluator, model, criterion, optimizer, scheduler, random_noise, ENV, label_name):
    beta1 = 0.5
    beta2 = 0.999
    lr = 0.001    
    num_domains = 1
    num_aug_domains = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    domain_generator = DomainGenerator(c_dim = 2 * num_domains).to(device)

    # load pretrained domain generator
    DomainGenerator_optimizer = torch.optim.Adam(domain_generator.parameters(), lr, (beta1, beta2))
    optimizer.add_param_group({'params': domain_generator.parameters()})
    # Generate Data loader
    datasets_generator = dataset.DatasetGenerator(args=args, train_batch_size=args.train_batch_size,
                                                  eval_batch_size=args.eval_batch_size,
                                                  train_data_type=args.train_data_type,
                                                  train_data_path=args.train_data_path,
                                                  test_data_type=args.test_data_type,
                                                  test_data_path=args.test_data_path,
                                                  num_of_workers=args.num_of_workers,
                                                  seed=args.seed, no_train_augments=True)

    index = 0
    lambda_domain = 1

    if args.use_subset:   
        data_loader = datasets_generator._split_validation_set(train_portion=args.universal_train_portion,
                                                               train_shuffle=True, train_drop_last=True)
    else:
        data_loader = datasets_generator.getDataLoader(train_shuffle=True, train_drop_last=True)

    condition = True
    logger.info('=' * 20 + 'Searching Universal Perturbation' + '=' * 20)
    if hasattr(model, 'classify'):
        model.classify = True

    # CLIP
    clip_model, preprocess = clip.load('ViT-B/32', device)
    with torch.no_grad():
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in label_name]).to(device)
        label_emb = clip_model.encode_text(text_inputs).float()

    for i in range(args.epoch):
        if args.attack_type == 'min-min' and not args.load_model:
            # Train Batch for min-min noise
            for j in range(0, args.train_step):
                try:
                    (images, labels) = next(data_iter)
                except:
                    data_iter = iter(data_loader[args.universal_train_target])
                    (images, labels) = next(data_iter)

                images, labels = images.to(device), labels.to(device)
                # =================================================================================== 
                #                                domain augmentation                                  #
                # =================================================================================== #
                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #
                fake=[]
                x_real = images
                label_org = torch.zeros(x_real.size(0),num_domains + num_aug_domains).cuda()
                label_org[:,index] = 1.0
                new_idx = num_domains + index
                label_trg = torch.zeros(x_real.size(0),num_domains + num_aug_domains).cuda()
                label_trg[:,new_idx] = 1.0

                if index == 0:
                    x_all = x_real
                    labels_all = labels
                else:
                    x_all = torch.cat((x_all,x_real),dim=0)
                    labels_all = torch.cat((labels_all,labels),dim=0)

                # =================================================================================== #
                #                               2. Train the domain composer                          #
                # =================================================================================== #

                domain_generator.train()
                x_fake = domain_generator(x_real)
                fake.append(x_fake)
                loss_aug = Loss_distribution(model, x_real, x_fake)

                total_lossG = -1 * lambda_domain*loss_aug
                total_lossG.backward()
                if (j+1) % 10 == 0:
                    logger.info('loss_aug: {:.4f} '.format(loss_aug.item()))

                DomainGenerator_optimizer.step()
                DomainGenerator_optimizer.zero_grad()

                label_fake = torch.zeros(x_real.size(0),num_domains + num_aug_domains).cuda()
                new_idx = num_domains + index 
                label_fake[:,new_idx] = 1.0
                fake = domain_generator(images)

                images = torch.cat((images, fake.detach()), 0)
                labels = torch.cat((labels, labels), 0)

                # Add Class-wise Noise to each sample
                train_imgs = []
                for i, (image, label) in enumerate(zip(images, labels)):
                    noise = random_noise[label.item()]  
                    mask_cord, class_noise = noise_generator._patch_noise_extend_to_img(noise, image_size=image.shape, patch_location=args.patch_location)
                    train_imgs.append(images[i]+class_noise)  
                # Train
                model.train()
                for param in model.parameters():
                    param.requires_grad = True
                log_payload = trainer.train_batch(torch.stack(train_imgs).to(device), labels, model, optimizer)
                loss_classifier = log_payload['loss']

        classwise_noise_all = [] 
        for i, (images, labels) in tqdm(enumerate(data_loader[args.universal_train_target]), total=len(data_loader[args.universal_train_target])):
            images, labels, model = images.to(device), labels.to(device), model.to(device)

            # Add Class-wise Noise to each sample
            batch_noise, mask_cord_list = [], []
            for i, (image, label) in enumerate(zip(images, labels)):
                noise = random_noise[label.item()]
                mask_cord, class_noise = noise_generator._patch_noise_extend_to_img(noise, image_size=image.shape, patch_location=args.patch_location)
                batch_noise.append(class_noise)
                mask_cord_list.append(mask_cord)

            batch_noise = torch.stack(batch_noise).to(device)  
            if args.attack_type == 'min-min':
                perturb_img, eta, loss_generator, loss_hinge, loss_plc = noise_generator.min_min_attack(args, images, labels, label_emb, model, clip_model, optimizer, criterion, random_noise=batch_noise)
            elif args.attack_type == 'min-max':
                perturb_img, eta = noise_generator.min_max_attack(images, labels, model, optimizer, criterion, random_noise=batch_noise)
            else:
                raise('Invalid attack')

            class_noise_eta = collections.defaultdict(list)
            for i in range(len(eta)):
                x1, x2, y1, y2 = mask_cord_list[i]
                delta = eta[i][:, x1: x2, y1: y2]
                class_noise_eta[labels[i].item()].append(delta.detach().cpu())

            currentbatch_noise = torch.zeros(*args.noise_shape)
            for key in class_noise_eta:
                currentbatch_noise[key] = torch.stack(class_noise_eta[key]).mean(dim=0)   
            classwise_noise_all.append(currentbatch_noise)

        classwise_noise_all = torch.stack(classwise_noise_all, dim=0)
        random_noise = torch.mean(classwise_noise_all, dim=0)

        # Eval termination conditions
        logger.info('loss_classifier: {:.4f} loss_generator: {:.4f} loss_hinge: {:.4f} loss_plc: {:.4f}'.format(loss_classifier.item(), loss_generator.item(), loss_hinge.item(), loss_plc.item()))
        random_noise = random_noise.detach()
        ENV['random_noise'] = random_noise
    return random_noise

def main():
    # Setup ENV
    datasets_generator = dataset.DatasetGenerator(args=args, train_batch_size=args.train_batch_size,
                                                  eval_batch_size=args.eval_batch_size,
                                                  train_data_type=args.train_data_type,
                                                  train_data_path=args.train_data_path,
                                                  test_data_type=args.test_data_type,
                                                  test_data_path=args.test_data_path,
                                                  num_of_workers=args.num_of_workers,
                                                  seed=args.seed)
    data_loader = datasets_generator.getDataLoader()

    if args.version=="vit":
        model = ViTCustom(img_size=args.image_size,num_classes=len(data_loader['train_dataset'].dataset.classes)).to(device)
    else: 
        model = config.model().to(device)

    model = config.model().to(device)
    logger.info("param size = %fMB", util.count_parameters_in_MB(model))
    optimizer = config.optimizer(model.parameters())
    scheduler = config.scheduler(optimizer)
    criterion = config.criterion()

    if args.train_data_type == 'CIFAR10' or args.train_data_type == 'CIFAR100':
        label_name = data_loader['train_dataset'].dataset.classes

    elif args.train_data_type == 'SVHN':
        label_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    if args.perturb_type == 'classwise':
        if args.use_subset:  
            data_loader = datasets_generator._split_validation_set(train_portion=args.universal_train_portion,
                                                                   train_shuffle=True, train_drop_last=True)
            train_target = 'train_subset'
        else:
            data_loader = datasets_generator.getDataLoader(train_shuffle=True, train_drop_last=True)
            train_target = 'train_dataset'

    trainer = Trainer(criterion, data_loader, logger, config, target=train_target)
    evaluator = Evaluator(data_loader, logger, config)
    ENV = {'global_step': 0,
           'best_acc': 0.0,
           'curren_acc': 0.0,
           'best_pgd_acc': 0.0,
           'train_history': [],
           'eval_history': [],
           'pgd_eval_history': [],
           'genotype_list': []}

    if args.data_parallel:  
        model = torch.nn.DataParallel(model)

    if args.load_model:  
        checkpoint = util.load_model(filename=checkpoint_path_file,
                                     model=model,
                                     optimizer=optimizer,
                                     alpha_optimizer=None,
                                     scheduler=scheduler)
        ENV = checkpoint['ENV']
        trainer.global_step = ENV['global_step']
        logger.info("File %s loaded!" % (checkpoint_path_file))

    noise_generator = toolbox.PerturbationTool(args,
                                               epsilon=args.epsilon,
                                               num_steps=args.num_steps,
                                               step_size=args.step_size)

    if args.attack_type == 'random':
        noise = noise_generator.random_noise(noise_shape=args.noise_shape)
        torch.save(noise, os.path.join(args.exp_name, 'perturbation.pt'))
        logger.info(noise)
        logger.info(noise.shape)
        logger.info('Noise saved at %s' % (os.path.join(args.exp_name, 'perturbation.pt')))
    elif args.attack_type == 'min-min' or args.attack_type == 'min-max':
        if args.attack_type == 'min-max':
            train(0, model, optimizer, scheduler, criterion, trainer, evaluator, ENV, data_loader)
        if args.random_start:  
            random_noise = noise_generator.random_noise(noise_shape=args.noise_shape)
        else:
            random_noise = torch.zeros(*args.noise_shape)   
        if args.perturb_type == 'classwise':
            noise = universal_perturbation(args, noise_generator, trainer, evaluator, model, criterion, optimizer, scheduler, random_noise, ENV, label_name)
        torch.save(noise, os.path.join(args.exp_name, 'perturbation.pt'))
        logger.info(noise)
        logger.info(noise.shape)
        logger.info('Noise saved at %s' % (os.path.join(args.exp_name, 'perturbation.pt')))

        # Save Model
        net_G = noise_generator.generator 
        filename = generator_path_file + '.pth'
        torch.save(net_G.state_dict(), filename)
        logger.info('Generator Saved at %s', filename)
    else:
        raise('Not implemented yet')
    return


if __name__ == '__main__':
    for arg in vars(args):
        logger.info("%s: %s" % (arg, getattr(args, arg)))
    start = time.time()
    main()
    end = time.time()
    cost = (end - start) / 86400
    payload = "Running Cost %.2f Days \n" % cost
    logger.info(payload)
