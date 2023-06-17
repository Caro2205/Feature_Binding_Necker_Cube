"""
Author: Kaltenberger
franziska.kaltenberger@student.uni-tuebingen.de
"""

from numpy.lib.function_base import angle
import torch 
import numpy as np
from torch._C import device
from torch.autograd import Variable

class Perspective_Taker():

    """
    Performs Perspective Taking.

    Initial parameters:
        - num_featrues: How many features does the internal model have? 
                        (i.e. number of input features of the coreLSTM)
        - num_dimensions: dimensionality number of input of internal model
                            corresponding to the Gestalt variants: 
                            > Pos: 3        -> 1-3: position
                            > PosVel: 6     -> 4-6: velocity
                            > PosDirMag: 7  -> 4-6: direction, 7: magnitude
    
    Important functions: 
        - rotate: rotates given input by means given rotation matrix 
        - qrotate: rotates given input by means of given quaternion
        - translate: translates given input by means of given translation bias
        - update functions of parameters are using 
            > SGD with given learning rate and momentum 
            > sign_damp=True: sign damping with given alpha
    """

    def __init__(self, num_features, num_dimensions):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.dimensions = num_dimensions
        self.num_features = num_features

        self.momentum_rotquat = torch.zeros(1, self.dimensions+1).to(self.device)
        self.momentum_roteul = torch.zeros(self.dimensions, 1).to(self.device)
        self.momentum_rotation = [None, None]
        self.momentum_trans = torch.zeros(1, self.dimensions).to(self.device)

        self.sign_damp_rotquat = torch.ones(self.dimensions+1).to(self.device)
        self.sign_damp_roteul = torch.ones(self.dimensions).to(self.device)
        self.sign_damp_translation = torch.ones(self.dimensions).to(self.device)


    def set_parameters(self, num_feat, num_dim):
        self.num_features = num_feat
        self.dimensions = num_dim


    #################################################################################
    #################### EULER ROTATION
    #################################################################################

    def init_angles_(self, init_axis_angle):
        q = self.init_quaternion(init_axis_angle)
        eul = torch.rad2deg(self.qeuler(q, 'xyz'))
        return eul.view(self.dimensions, 1)

    
    def compute_rotation_matrix_(self, alpha, beta, gamma):
        alpha_rad = torch.deg2rad(alpha)
        beta_rad = torch.deg2rad(beta)
        gamma_rad = torch.deg2rad(gamma)

        # initialize dimensional rotation matrices 
        R_x_1 = torch.Tensor([[1.0,0.0,0.0]])
        R_x_2 = torch.stack([torch.zeros(1), torch.cos(alpha_rad), - torch.sin(alpha_rad)], dim=1)
        R_x_3 = torch.stack([torch.zeros(1), torch.sin(alpha_rad), torch.cos(alpha_rad)], dim=1)
        R_x = torch.stack([R_x_1, R_x_2, R_x_3], dim=1)
        
        R_y_1 = torch.stack([torch.cos(beta_rad), torch.zeros(1), torch.sin(beta_rad)], dim=1)
        R_y_2 = torch.Tensor([[0.0,1.0,0.0]])
        R_y_3 = torch.stack([- torch.sin(beta_rad), torch.zeros(1), torch.cos(beta_rad)], dim=1)
        R_y = torch.stack([R_y_1, R_y_2, R_y_3], dim=1)

        R_z_1 = torch.stack([torch.cos(gamma_rad), - torch.sin(gamma_rad), torch.zeros(1)], dim=1)
        R_z_2 = torch.stack([torch.sin(gamma_rad), torch.cos(gamma_rad), torch.zeros(1)], dim=1)
        R_z_3 = torch.Tensor([[0.0,0.0,1.0]])
        R_z = torch.stack([R_z_1, R_z_2, R_z_3], dim=1)

        # compute rotation matrix
        rotation_matrix = torch.matmul(R_x, torch.matmul(R_y, R_z))    

        return rotation_matrix


    def update_rotation_angles_(self, rotation_angles, gradient, learning_rate, momentum):
        rotation_angles = torch.stack(rotation_angles)

        mom = momentum*self.momentum_roteul
        # update with gradient descent
        upd_rotation_angles = []
        for i in range(self.dimensions):
            with torch.no_grad():
                e = - learning_rate * gradient[i] + mom[i]
                upd_rotation_angles.append(e)

        self.momentum_roteul = torch.stack(upd_rotation_angles)        
                
        return rotation_angles + self.momentum_roteul


    def rotate(self, input, rotation_matrix):
        return torch.matmul(rotation_matrix, input.T).T


    #################################################################################
    #################### QUATERNION ROTATION
    #################################################################################

    def init_quaternion(self, init_axis_angle): 

        q = torch.zeros(1,4)
        if init_axis_angle == 0:
            # init with axis angle of 0°
            q[0,0] = 1.0

        elif init_axis_angle == 35:
            # init with axis angle of 35°
            q[0,0] = 0.953717 
            q[0,1] = 0.1736126
            q[0,2] = 0.1736126
            q[0,3] = 0.1736126

        elif init_axis_angle == 90:
            # init with axis angle of 90°
            q[0,0] = 0.7071068 
            q[0,1] = 0.4082483 
            q[0,2] = 0.4082483 
            q[0,3] = 0.4082483 

        elif init_axis_angle == 45:
            # init with axis angle of 45°
            q[0,0] = 0.9238795  
            q[0,1] = 0.2209424 
            q[0,2] = 0.2209424 
            q[0,3] = 0.2209424 

        elif init_axis_angle == 135:
            # init with axis angle of 135°
            q[0,0] = 0.3826834  
            q[0,1] = 0.5334021 
            q[0,2] = 0.5334021 
            q[0,3] = 0.5334021

        elif init_axis_angle == 180:
            # init with axis angle of 180°
            q[0,0] = 0.0
            q[0,1] = 0.5773503 
            q[0,2] = 0.5773503 
            q[0,3] = 0.5773503

        else: 
            # invalid axis angle
            print(f'Received invalid initial axis angle: {init_axis_angle}')
            exit()

        q = self.norm_quaternion(q)

        return q


    def norm_quaternion(self, q): 
        abs = torch.sqrt(torch.sum(torch.mul(q,q)))
        return torch.div(q, abs)

    
    def update_quaternion(self, q, gradient, learning_rate, momentum, sign_damp=False, alpha=0.0):
        # calculate momentum term 
        mom = momentum*self.momentum_rotquat

        # calculate change in rotation quaternion
        if sign_damp: 
            # update sign damping variable
            self.sign_damp_rotquat = alpha * self.sign_damp_rotquat + (1-alpha) * torch.sign(gradient)
            upd = - learning_rate * gradient * torch.square(self.sign_damp_rotquat) + mom
        else:
            upd = - learning_rate * gradient + mom

        # reset momentum
        self.momentum_rotquat = upd
        
        return self.norm_quaternion(q + upd)


    """
    (Pavllo et al., 2018)
    """
    def qeuler(self, q, order, epsilon=0):
        """
        Convert quaternion(s) q to Euler angles.
        Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
        Returns a tensor of shape (*, 3).
        """
        assert q.shape[-1] == 4
        
        original_shape = list(q.shape)
        original_shape[-1] = 3
        q = q.view(-1, 4)
        
        q0 = q[:, 0]
        q1 = q[:, 1]
        q2 = q[:, 2]
        q3 = q[:, 3]


        if order == 'xyz':
            x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2*(q1 * q1 + q2 * q2))
            y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1+epsilon, 1-epsilon))
            z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))
        elif order == 'yzx':
            x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2*(q1 * q1 + q3 * q3))
            y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2*(q2 * q2 + q3 * q3))
            z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1+epsilon, 1-epsilon))
        elif order == 'zxy':
            x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1+epsilon, 1-epsilon))
            y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2*(q1 * q1 + q2 * q2))
            z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q1 * q1 + q3 * q3))
        elif order == 'xzy':
            x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2*(q1 * q1 + q3 * q3))
            y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2*(q2 * q2 + q3 * q3))
            z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1+epsilon, 1-epsilon))
        elif order == 'yxz':
            x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1+epsilon, 1-epsilon))
            y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2*(q1 * q1 + q2 * q2))
            z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2*(q1 * q1 + q3 * q3))
        elif order == 'zyx':
            x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2*(q1 * q1 + q2 * q2))
            y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1+epsilon, 1-epsilon))
            z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))
        else:
            raise

        return torch.stack((x, y, z), dim=1).view(original_shape)


    def quaternion2rotmat(self, q):
        if q.size()[0] == 1: 
            q = q[0]

        # get single quaternion entries 
        w = q[0]
        x = q[1]
        y = q[2]
        z = q[3]

        # multiplied entries
        ww = w*w
        wx = w*x
        wy = w*y
        wz = w*z
        
        xx = x*x
        xy = x*y
        xz = x*z

        yy = y*y
        yz = y*z

        zz = z*z


        rotation_matrix = torch.Tensor(
            [[2*(ww+xx)-1,   2*(xy-wz),   2*(xz+wy)], 
             [  2*(xy+wz), 2*(ww+yy)-1,   2*(yz-wx)], 
             [  2*(xz-wy),   2*(yz+wx), 2*(ww+zz)-1]]
        ).to(self.device)

        return rotation_matrix


    """
    (Pavllo et al., 2018)
    """
    def qrotate(self, v, q):
        """
        Rotate vector(s) v about the rotation described by quaternion(s) q.
        Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
        where * denotes any number of dimensions.
        Returns a tensor of shape (*, 3).
        """
        original_shape = list(v.shape)

        q = torch.stack([q] * original_shape[0])

        q = q.view(-1, 4)
        v = v.view(-1, 3)

        qvec = q[:, 1:]
        uv = torch.cross(qvec, v, dim=1)
        uuv = torch.cross(qvec, uv, dim=1)
        return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


    """
    (Pavllo et al., 2018)
    """
    def qmul(self, q, r):
        """
        Multiply quaternion(s) q with quaternion(s) r.
        Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
        Returns q*r as a tensor of shape (*, 4).
        """
        assert q.shape[-1] == 4
        assert r.shape[-1] == 4
        
        original_shape = q.shape
        
        # Compute outer product
        terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

        w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
        x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
        y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
        z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
        return torch.stack((w, x, y, z), dim=1).view(original_shape)


    def inverse_rotation_quaternion(self, q): 
        original_shape = q.shape
        q = q.view(4)
        q_inv = torch.zeros(4)
        q_inv[0] = q[0]

        for i in range(3):
            q_inv[i+1] = -q[i+1]

        q_inv = q_inv.view(original_shape)
        abs = torch.sum(torch.mul(q,q))
        q_inv = torch.div(q_inv, abs)

        return q_inv


    def negate_quaternion(self, q): 
        return q * -1


    #################################################################################
    #################### TRANSLATION
    #################################################################################


    def init_translation_bias_(self):
        return torch.zeros(self.dimensions)

   
    def update_translation_bias_(self, translation_bias, gradient, learning_rate, momentum, sign_damp=False, alpha=0):
        # calculate momentum term 
        mom = momentum*self.momentum_trans

        # calculate change in translation bias
        if sign_damp: 
            # update sign damping variable
            self.sign_damp_translation = alpha * self.sign_damp_translation + (1-alpha) * torch.sign(gradient)

            upd = - learning_rate * gradient * torch.square(self.sign_damp_translation) + mom
        else: 
            upd = - learning_rate * gradient + mom

        # reset momentum
        self.momentum_trans = upd

        # update translation bias 
        return translation_bias + upd


    def translate(self, input, translation_bias): 
        return input + translation_bias
    

    def inverse_translation_bias(self, t): 
        return t * -1

