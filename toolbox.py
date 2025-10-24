import collections
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from models.GAN import Generator
import clip
from PIL import Image
from torchvision import transforms

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class PerturbationTool():
    def __init__(self, args, seed=0, epsilon=0.03137254901, num_steps=20, step_size=0.00784313725):
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.seed = seed
        self.generator = Generator(args).cuda()
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.001)
        np.random.seed(seed)

    def random_noise(self, noise_shape=[10, 3, 32, 32]):
        random_noise = torch.FloatTensor(*noise_shape).uniform_(-self.epsilon, self.epsilon).to(device)
        return random_noise

    def min_min_attack(self, args, images, labels, label_emb, model, clip_model, optimizer, criterion, random_noise=None, sample_wise=False):
        for _ in range(self.num_steps):
            noise = self.generator(images)    
            temp_class_noise = collections.defaultdict(list)
            currentbatch_class_noise = torch.zeros(*args.noise_shape)
            for i in range(len(noise)):
                temp_class_noise[labels[i].item()].append(noise[i])
            for key in temp_class_noise:
                currentbatch_class_noise[key] = torch.stack(temp_class_noise[key]).mean(dim=0)

            noise = torch.stack([currentbatch_class_noise[label.item()] for label in labels]).cuda()

            perturb_img = torch.clamp(images + noise, 0, 1)
                
            # use fixed surrogate model to get noise embedding
            noise_emb, _ = model(noise)  
            sim = torch.mm(noise_emb, label_emb.t().detach())
            sim = F.log_softmax(sim, dim=1)
            
            # PLC loss
            batch_size = sim.size(0)
            loss_plc = 0
            for i in range(batch_size):
                loss_plc += sim[i, labels[i]]
            loss_plc = -loss_plc / batch_size  

            # CE loss of generator
            _, logits = model(perturb_img)
            loss_G_classification = criterion(logits, labels).requires_grad_()

            # Hinge loss
            noise_bound = args.epsilon
            noise_norm = torch.mean(torch.norm(noise.view(noise.shape[0], -1), float('inf'), dim=1))
            total_loss_perturb_norm = torch.max(noise_norm - noise_bound, torch.zeros(1).cuda())

            loss_G = loss_G_classification + 2*total_loss_perturb_norm  + loss_plc

            self.optimizer_G.zero_grad()
            loss_G.backward()
            self.optimizer_G.step()

        return perturb_img, noise, loss_G_classification, total_loss_perturb_norm, loss_plc

    def min_max_attack(self, images, labels, model, optimizer, criterion, random_noise=None, sample_wise=False):
        if random_noise is None:
            random_noise = torch.FloatTensor(*images.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        perturb_img = Variable(images.data + random_noise, requires_grad=True)
        perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)
        eta = random_noise
        for _ in range(self.num_steps):
            opt = torch.optim.SGD([perturb_img], lr=1e-3)
            opt.zero_grad()
            model.zero_grad()
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                logits = model(perturb_img)
                loss = criterion(logits, labels)
            else:
                logits, loss = criterion(model, perturb_img, labels, optimizer)
            loss.backward()

            eta = self.step_size * perturb_img.grad.data.sign()
            perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
            eta = torch.clamp(perturb_img.data - images.data, -self.epsilon, self.epsilon)
            perturb_img = Variable(images.data + eta, requires_grad=True)
            perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)

        return perturb_img, eta

    def _patch_noise_extend_to_img(self, noise, image_size=[3, 32, 32], patch_location='center'):
        c, h, w = image_size[0], image_size[1], image_size[2]
        mask = np.zeros((c, h, w), np.float32)
        x_len, y_len = noise.shape[1], noise.shape[1]

        if patch_location == 'center' or (h == w == x_len == y_len):
            x = h // 2
            y = w // 2
        elif patch_location == 'random':
            x = np.random.randint(x_len // 2, w - x_len // 2)
            y = np.random.randint(y_len // 2, h - y_len // 2)
        else:
            raise('Invalid patch location')

        x1 = np.clip(x - x_len // 2, 0, h)
        x2 = np.clip(x + x_len // 2, 0, h)
        y1 = np.clip(y - y_len // 2, 0, w)
        y2 = np.clip(y + y_len // 2, 0, w)
        if type(noise) is np.ndarray:
            pass
        else:
            mask[:, x1: x2, y1: y2] = noise.cpu().numpy()
        return ((x1, x2, y1, y2), torch.from_numpy(mask).to(device))
