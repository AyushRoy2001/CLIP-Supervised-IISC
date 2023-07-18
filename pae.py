import torch
from os.path import join as pjoin

class Ortho:
    def _init_(self):
        super(Ortho,self).__init__()
        #pass

    def getOrtho(self, vect_space_org):
        """
        Performs QR decomposition of vectors

        Parameters:
        vect[][]: 2D array with rows as vectors to be orthonormalised
        """
        vect_space = vect_space_org.clone().detach()
        
        for i in range(vect_space.shape[0]):
            for j in range(i):
                vect_space[i] -= torch.dot(vect_space[i], vect_space[j])*vect_space[j]
            norm = torch.sqrt(torch.dot(vect_space[i], vect_space[i]))
            if norm>0.:
                vect_space[i] /= norm
        return vect_space
