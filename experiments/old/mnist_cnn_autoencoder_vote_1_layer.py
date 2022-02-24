import rusty_neat
from rusty_neat import ndalgebra as nd
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


# If you get 503 while downloading MNIST then download it manually
# wget www.di.ens.fr/~lelarge/MNIST.tar.gz
# tar -zxvf MNIST.tar.gz


class k_winners(torch.autograd.Function):
    """
    A K-winner take all autograd function for CNN 2D inputs (batch, Channel, H, W).
    .. seealso::
         Function :class:`k_winners`
    """

    @staticmethod
    def forward(ctx, x, dutyCycles, k, boostStrength):
        """
        Use the boost strength to compute a boost factor for each unit represented
        in x. These factors are used to increase the impact of each unit to improve
        their chances of being chosen. This encourages participation of more columns
        in the learning process. See :meth:`k_winners.forward` for more details.
        :param ctx:
          Place where we can store information we will need to compute the gradients
          for the backward pass.
        :param x:
          Current activity of each unit.
        :param dutyCycles:
          The averaged duty cycle of each unit.
        :param k:
          The activity of the top k units will be allowed to remain, the rest are
          set to zero.
        :param boostStrength:
          A boost strength of 0.0 has no effect on x.
        :return:
          A tensor representing the activity of x after k-winner take all.
        """
        if boostStrength > 0.0:
            targetDensity = float(k) / x.shape[1]
            boostFactors = torch.exp((targetDensity - dutyCycles) * boostStrength)
            boosted = x.detach() * boostFactors
        else:
            boosted = x.detach()

        # Take the boosted version of the input x, find the top k winners.
        # Compute an output that only contains the values of x corresponding to the top k
        # boosted values. The rest of the elements in the output should be 0.
        res = torch.zeros_like(boosted)
        topk, indices = boosted.topk(k, dim=1, sorted=False)
        #####################################################################################
        res.scatter_(1, indices, 1)  # <--- this right here! This is basically the only difference
        # between SAE and SAE_STE. We set 1 instead of original value from x.
        # This iwa the output is binary instead of continuous
        #####################################################################################

        ctx.save_for_backward(indices)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass, we set the gradient to 1 for the winning units, and 0
        for the others.
        """
        indices, = ctx.saved_tensors

        grad_x = torch.zeros_like(grad_output, requires_grad=False)
        grad_x.scatter_(1, indices, grad_output.gather(1, indices))

        return grad_x, None, None, None


class KWinners(nn.Module):
    """
    Applies K-Winner function to the input tensor
    See :class:`htmresearch.frameworks.pytorch.functions.k_winners2d`
    """

    def __init__(self, n, k, kInferenceFactor=1.0, boostStrength=1.0,
                 boostStrengthFactor=1.0, dutyCyclePeriod=1000):
        """
        :param n:
          Number of units. Usually the output of the max pool or whichever layer
          preceding the KWinners2d layer.
        :type n: int
        :param k:
          The activity of the top k units will be allowed to remain, the rest are set
          to zero
        :type k: int
        :param kInferenceFactor:
          During inference (training=False) we increase k by this factor.
        :type kInferenceFactor: float
        :param boostStrength:
          boost strength (0.0 implies no boosting).
        :type boostStrength: float
        :param boostStrengthFactor:
          Boost strength factor to use [0..1]
        :type boostStrengthFactor: float
        :param dutyCyclePeriod:
          The period used to calculate duty cycles
        :type dutyCyclePeriod: int
        """
        super().__init__()
        assert (boostStrength >= 0.0)

        self.n = n
        self.k = k
        self.kInferenceFactor = kInferenceFactor
        self.learningIterations = 0

        # Boosting related parameters
        self.boostStrength = boostStrength
        self.boostStrengthFactor = boostStrengthFactor
        self.dutyCyclePeriod = dutyCyclePeriod

        self.register_buffer("dutyCycle", torch.zeros(n))

    def forward(self, x):
        # Apply k-winner algorithm if k < n, otherwise default to standard RELU
        if self.k >= self.n:
            return F.relu(x)

        if self.training:
            k = self.k
        else:
            k = min(int(round(self.k * self.kInferenceFactor)), self.n)

        x = k_winners.apply(x, self.dutyCycle, k, self.boostStrength)

        if self.training:
            self.updateDutyCycle(x)

        return x

    def updateDutyCycle(self, x):
        batchSize = x.shape[0]
        self.learningIterations += batchSize
        period = min(self.dutyCyclePeriod, self.learningIterations)
        self.dutyCycle.mul_(period - batchSize)
        self.dutyCycle.add_(x.gt(0).sum(dim=0, dtype=torch.float))
        self.dutyCycle.div_(period)

    def entropy(self):
        """
        Returns the current total entropy of this layer
        """
        if self.k < self.n:
            """
            Calculate entropy for a list of binary random variables
            :param x: (torch tensor) the probability of the variable to be 1.
            :return: entropy: (torch tensor) entropy, sum(entropy)
            """
            x = self.dutyCycle
            entropy = - x * x.log2() - (1 - x) * (1 - x).log2()
            entropy[x * (1 - x) == 0] = 0
            return entropy.sum()
        else:
            return 0


class Autoencoder(nn.Module):

    def __init__(self, width, height, bottleneck, k, boost_strength):
        self.width = width
        self.height = height
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=5)
        self.conv2 = nn.Conv2d(2, 4, kernel_size=5)
        self.conv3 = nn.Conv2d(4, 8, kernel_size=5)
        self.hidden_size = (self.width - 4 * 3) * (self.height - 4 * 3) * 8
        self.lin1 = nn.Linear(self.hidden_size, bottleneck)
        self.k_winners = KWinners(bottleneck, k, boostStrength=boost_strength)
        self.lin2 = nn.Linear(bottleneck, 10)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.conv1(x)
        x = F.relu(x, True)
        x = self.conv2(x)
        x = F.relu(x, True)
        x = self.conv3(x)
        x = F.relu(x, True)
        x = x.view(batch_size, self.hidden_size)
        x = self.lin1(x)
        x = self.k_winners(x)
        return x


# Defining Parameters
BATCH_SIZE = 1

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
trainset = tv.datasets.MNIST('.', train=True, download=True, transform=transform)
testset = tv.datasets.MNIST('.', train=False, transform=transform)
train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

EPOCHS = 1000
BITS = 1024
model = Autoencoder(28, 28, BITS, 32, 1.0).to(DEVICE)
model.train(False)
model.load_state_dict(torch.load('sc_ste.pth'))
torch.autograd.set_grad_enabled(False)

inp_enc = rusty_neat.htm.EncoderBuilder()
img_enc = inp_enc.add_bits(BITS)
out_enc = rusty_neat.htm.EncoderBuilder()
lbl_enc = out_enc.add_categorical(10, 128)
bitset = rusty_neat.htm.CpuBitset(inp_enc.input_size)
active_columns_bitset = rusty_neat.htm.CpuBitset(out_enc.input_size)
htm = None


def infer(img, lbl=None):
    bitset.clear()
    encoded_bits = model(img)
    img_enc.encode(bitset, encoded_bits.type(torch.bool).squeeze(0).tolist())
    predicted_columns = htm.compute(bitset)
    if lbl is not None:
        lbl_enc.encode(active_columns_bitset, lbl)
        htm.update_permanence(predicted_columns, active_columns_bitset, bitset)
        active_columns_bitset.clear()
    else:
        return lbl_enc.find_category_with_highest_overlap(predicted_columns)


def train(repetitions, c):
    for _ in range(repetitions):
        for i, (img, lbl) in tqdm(enumerate(train_loader), desc="training", total=c):
            infer(img, lbl)
            if i == c:
                break


def eval(c):
    confusion_matrix = np.zeros((10, 10))
    for i, (img, lbl) in tqdm(enumerate(test_loader), desc="evaluation", total=len(test_loader)):
        guessed = infer(img)
        confusion_matrix[guessed, lbl] += 1
        if i == c:
            break
    return confusion_matrix


def random_trials(repetitions, trials, samples, test_samples):
    global htm
    confusion_matrix_avg = np.zeros((10, 10))
    for _ in tqdm(range(trials), desc="trial", total=trials):
        htm = rusty_neat.htm.CpuHTM2(inp_enc.input_size, out_enc.input_size, lbl_enc.sdr_cardinality,
                                     int(inp_enc.input_size*0.8))
        train(repetitions, samples)
        correct = eval(test_samples)
        print(sum(correct.diagonal()), "/", test_samples, "=", sum(correct.diagonal()) / test_samples)
        confusion_matrix_avg += correct
    return confusion_matrix_avg / (trials * test_samples)


def run(repetitions, trials, samples, test_samples):
    acc = random_trials(repetitions, trials, samples, test_samples)
    print("Ensemble accuracy(" + str(repetitions) + "," + str(trials) + "," + str(samples) + "," + str(test_samples) + "):",
          sum(acc.diagonal()))
    print(acc)

# htm = rusty_neat.htm.CpuHTM2(inp_enc.input_size, out_enc.input_size, lbl_enc.sdr_cardinality,
#                                      int(inp_enc.input_size*0.8))
# htm.update_permanence_ltd(predicted_columns, active_columns_bitset, bitset)
# Ensemble accuracy(2, 20, 1000, 1000):  0.1558

# htm = rusty_neat.htm.CpuHTM4(inp_enc.input_size, out_enc.input_size, lbl_enc.sdr_cardinality,
#                                      int(inp_enc.input_size), 0.5)
# htm.update_permanence_ltd(predicted_columns, active_columns_bitset, bitset)
# Ensemble accuracy(2, 20, 1000, 1000):  0.636599

run(2, 20, 1000, 1000)
