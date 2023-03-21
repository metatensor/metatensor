import equistore
from equistore import TensorMap, TensorBlock, Labels
import numpy as np
print(equistore.__file__, flush=True)

class JoinSuite:
    def setup(self):
        # PR COMMENT
        #  this file is run in some tmp folder so 
        #  I download it for now to simplify life
        #  not intended to be merged
        import wget
        wget.download("https://github.com/lab-cosmo/equistore/raw/master/python/tests/data/qm7-power-spectrum.npz")
        self.tensor = equistore.load(
            "qm7-power-spectrum.npz",
            use_numpy=True,
        )

    def time_equistore_join(self):
        equistore.join([self.tensor, self.tensor], axis="properties")
