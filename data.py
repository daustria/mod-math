from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np

def write_in_base(x, int_len, base):
    """
    Writes x, an iterable of integer(s), in base. Pads the most significant bits with zeros. 
    E.g., if base = 8, p = 251, for x = array([10]) it would return array([0, 1, 2]).
    """
    results = []
    for x0 in x:
        result = np.ones(int_len) * x0
        result = (result // base ** np.arange(int_len)) % base
        results.append(result.astype(int)[::-1])
    return np.concatenate(results)

class ModularMult(object):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.rng = np.random.RandomState(params.seed)
        self.test_data = self.rng.choice(params.p, params.test_size, replace=False)
        self.base, self.int_len = params.base, params.int_len
        self.p = params.p

    def get_sample(self, x):
        y = (x * self.params.s) % self.p
        x_ = write_in_base([x], self.int_len, self.base)
        y_ = write_in_base([y], self.int_len, self.base)
        return x_, y_

    def gen_train_x(self):
        return self.rng.randint(self.p)

class TrainSet(Dataset):
    def __init__(self, generator, max_train_size=1000):
        super().__init__()
        self.generator = generator
        self.params = generator.params
        self.test_set = set(generator.test_data)
        self.max_train_size = max_train_size

    def __len__(self):
        return min(self.params.p - self.params.test_size, self.max_train_size)

    def __getitem__(self, index):
        while True:
            x = self.generator.gen_train_x()
            if x in self.test_set:
                continue
            sample_x, sample_y = self.generator.get_sample(x)
            padded_sample_y = np.insert(sample_y,0, self.params.start_token)
            return sample_x, padded_sample_y

class TestSet(Dataset):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator
        self.params = generator.params
        self.test_data = generator.test_data

    def __len__(self):
        return self.params.test_size

    def __getitem__(self, index):
        x = self.test_data[self.generator.rng.randint(self.params.test_size)]
        sample_x, sample_y = self.generator.get_sample(x)
        padded_sample_y = np.insert(sample_y,0, self.params.start_token)
        return sample_x, padded_sample_y

def create_datasets(params, max_train_size=1000):
    gen_obj = {
            'modularmult': ModularMult
            # 'modularmult2': ModularMult2, 
            # 'dlp': DLP, 
            # 'diffiehellman': DiffieHellman,
            # 'diffiehellmanfixed': DiffieHellmanFixed
            }
    gen = gen_obj['modularmult'](params)
    train_dataset = TrainSet(gen, max_train_size)
    test_dataset = TestSet(gen)
    return train_dataset, test_dataset
