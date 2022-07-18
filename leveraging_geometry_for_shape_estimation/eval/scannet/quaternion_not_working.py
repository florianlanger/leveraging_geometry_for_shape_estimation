# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Quaternion provides a class for manipulating quaternion objects.  This class provides:

  - convenient ways to deal with rotation representations (equatorial
    coordinates, matrix and quaternion):

    - a constructor to initialize from rotations in various representations,
    - conversion methods to the different representations.

  - methods to multiply and divide quaternions.

:Copyright: Smithsonian Astrophysical Observatory (2010)
:Authors: - Tom Aldcroft (aldcroft@cfa.harvard.edu)
          - Jean Connelly (jconnelly@cfa.harvard.edu)
          - Javier Gonzalez (javier.gonzalez@cfa.harvard.edu)
"""
# Copyright (c) 2010, Smithsonian Astrophysical Observatory
# All rights reserved.
##
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
# * Neither the name of the Smithsonian Astrophysical Observatory nor the
# names of its contributors may be used to endorse or promote products
# derived from this software without specific prior written permission.
##
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import operator
import numpy as np
from astropy.utils.shapes import ShapedLikeNDArray


class Quat(ShapedLikeNDArray):
    # None of the base class method functions are defined for quaternions.
    # These include things like min, max, round, all, etc. Calling these numpy
    # functions on Quat just goes to the numpy function, which may or may not
    # do something or else give an exception.
    _METHOD_FUNCTIONS = {}

    """
    Quaternion class

    Example usage::

     >>> from Quaternion import Quat
     >>> quat = Quat(equatorial=(12,45,45))
     >>> quat.ra, quat.dec, quat.roll
     (12, 45, 45)
     >>> quat.q
     array([ 0.38857298, -0.3146602 ,  0.23486498,  0.8335697 ])
     >>> q2 = Quat(q=quat.q)
     >>> q2.ra
     12.0

    Multiplication and division operators are overloaded for the class to
    perform appropriate quaternion multiplication and division.

    Quaternion composition as a multiplication q = q1 * q2 is equivalent to
    applying the q2 transform followed by the q1 transform.  Another way to
    express this is::

      q = Quat(transform=q1.transform @ q2.transform)

    Example usage::

      >>> q1 = Quat(equatorial=(20, 30, 0))
      >>> q2 = Quat(equatorial=(0, 0, 40))
      >>> (q1 * q2).equatorial
      array([20., 30., 40.])

    This example first rolls about X by 40 degrees, then rotates that rolled frame
    to RA=20 and Dec=30.  Doing the composition in the other order does a roll about
    (the original) X-axis of the (RA, Dec) = (20, 30) frame, yielding a non-intuitive
    though correct result::

      >>> (q2 * q1).equatorial
      array([ 353.37684725,   34.98868888,   47.499696  ])

    Note that each step is as described in the section
    :ref:`Equatorial -> Matrix <equatorialmatrix>`

      >>> q1 = Quat(equatorial=(20, 0, 0))
      >>> q2 = Quat(equatorial=(0, 30, 0))
      >>> q3 = Quat(equatorial=(0, 0, 40))
      >>> q1.transform, q2.transform, q3.transform
      (array([[ 0.93969262, -0.34202014,  0.        ],
              [ 0.34202014,  0.93969262, -0.        ],
              [ 0.        ,  0.        ,  1.        ]]),
       array([[ 0.8660254, -0.       , -0.5      ],
              [ 0.       ,  1.       , -0.       ],
              [ 0.5      ,  0.       ,  0.8660254]]),
       array([[ 1.        , -0.        ,  0.        ],
              [ 0.        ,  0.76604444, -0.64278761],
              [ 0.        ,  0.64278761,  0.76604444]]))

    Be aware that for Chandra operations the correct formalism for applying a
    delta quaternion ``dq`` to maneuver from ``q1`` to ``q2`` is::

      >>> q2 = q1 * dq
      >>> dq = q1.inv() * q2

    The quaternion returned by ``q1 / q2`` using the divide operator represents the
    delta quaternion in the INERTIAL FRAME, which is not what is used by the
    spacecraft to represent maneuvers.

    Instead use the ``dq`` method (or equivalently ``q1.inv() * q2``) for computing a
    delta quaternion to maneuver from ``q1`` to ``q2``::

      >>> dq = q1.dq(q2)

    When dealing with a collection of quaternions, it is much faster to operate
    on all of them at a time::

      >>> eq1 = [[ 30.000008,  39.999999,  49.999995],
      ...        [ 30.000016,  39.999999,  49.99999 ],
      ...        [ 30.000024,  39.999998,  49.999984],
      ...        [ 30.000032,  39.999998,  49.999979],
      ...        [ 30.00004 ,  39.999997,  49.999974]])
      >>> eq2 = [[ 34.679189,  42.454957,  43.647651],
      ...        [ 26.457503,  58.05772 ,  70.638782],
      ...        [ 42.283086,  50.072731,  36.676534],
      ...        [ 27.138448,  31.389365,  34.429049],
      ...        [ 22.722112,  37.893094,  50.06074 ]]
      >>> q1 = Quat(equatorial=eq1)
      >>> q2 = Quat(equatorial=eq2)
      >>> dq = q1.dq(q2)
      >>> dq.q
      array([[-0.02848482,  0.00774419,  0.03661687,  0.99889331],
             [ 0.15387087, -0.09527764,  0.12624189,  0.97535066],
             [-0.03970954, -0.01159734,  0.11488968,  0.99251651],
             [-0.14947438,  0.04195765, -0.06544431,  0.98570483],
             [-0.0393673 , -0.02604388, -0.045771  ,  0.99783613]])

    :param attitude: initialization attitude for quat

    ``attitude`` may be:
      * another Quat
      * a 4 element array (expects x,y,z,w quat form)
      * a 3 element array (expects ra,dec,roll in degrees)
      * a 3x3 transform/rotation matrix

    :param q: attitude as a quaternion.

    ``q`` must be an array with shape (4,) or (N, 4), with arbitrary N.
    The last axis corresponds to quaternion coordinates x, y, z, w.
    For example: (3, 2, 4) corresponds to a (3, 2) array of quaternions.

    :param equatorial: attitude in equatorial coordinates.

    ``q`` must be an array with shape (3,) or (N, 3), with arbitrary N.
    The last axis corresponds to equatorial coordinates ra, dec, roll.
    For example: (3, 2, 3) corresponds to a (3, 2) array of quaternions.

    :param transform: attitude as a 3x3 transform.

    ``q`` must be an array with shape (3, 3) or (N, 3, 3), with arbitrary N.
    The last two axes correspond to (3, 3) transformations.
    For example: (3, 2, 3, 3) corresponds to a (3, 2) array of quaternions.
    """

    def __new__(cls, attitude=None, transform=None, q=None, equatorial=None):
        self = super().__new__(cls)

        self._shape = None
        self._q = None
        self._equatorial = None
        self._T = None
        # other data members that are set lazily.
        self._ra0 = None
        self._roll0 = None
        return self

    def __init__(self, attitude=None, transform=None, q=None, equatorial=None):
        npar = (int(attitude is not None) + int(transform is not None)
                + int(q is not None) + int(equatorial is not None))
        if npar != 1:
            raise ValueError(
                f'{npar} arguments passed to constructor that takes only one of'
                ' attitude, transform, quaternion, equatorial.'
            )

        # checks to see if we've been passed a Quat
        if isinstance(attitude, Quat):
            q = attitude.q
        elif attitude is not None:
            # check to see if it is a supported shape
            attitude = np.array(attitude, dtype=np.float64)
            if attitude.shape == (4,):
                q = attitude
            elif attitude.shape == (3, 3):
                transform = attitude
            elif attitude.shape == (3,):
                equatorial = attitude
            else:
                try:
                    shape = attitude.shape
                    shape = f' (shape {shape})'
                except Exception:
                    shape = ''
                raise TypeError(
                    f"attitude argument{shape} is not one of an allowed type:"
                    " Quat or array with shape (...,3), (...,4), or (..., 3, 3)")

        # checking correct shapes
        if q is not None:
            q = np.atleast_1d(q).astype(np.float64)
            self._shape = q.shape[:-1]
            if q.shape[-1:] != (4,):
                raise TypeError("Creating a Quaternion from quaternion(s) "
                                "requires shape (..., 4), not {}".format(q.shape))
            self.q = q
        elif transform is not None:
            transform = np.atleast_2d(transform).astype(np.float64)
            self._shape = transform.shape[:-2]
            if transform.shape[-2:] != (3, 3):
                raise TypeError("Creating a Quaternion from quaternion(s) "
                                "requires shape (..., 3, 3), not {}".format(transform.shape))
            self.transform = transform
        elif equatorial is not None:
            equatorial = np.atleast_1d(equatorial).astype(np.float64)
            self._shape = equatorial.shape[:-1]
            if equatorial.shape[-1:] != (3,):
                raise TypeError("Creating a Quaternion from ra, dec, roll "
                                "requires shape (..., 3), not {}".format(equatorial.shape))
            self.equatorial = equatorial
        assert self._shape is not None

    @property
    def shape(self):
        """
        The shape of the multi-quaternion.

        For example, if the Quat is:
        - a single quaternion, then its shape is ().
        - a multi-quaternion with self.q.shape = (N, 4), then its shape is (N,)

        :returns: self.q.shape[:-1]
        :rtype: tuple
        """
        return self._shape

    @property
    def q(self):
        """
        Retrieve 4-vector of quaternion elements in [x, y, z, w] form
        or N x 4-vector if N > 1.

        :rtype: numpy array

        """
        if self._q is None:
            # Figure out q from available values, doing nothing others are not defined
            if self._equatorial is not None:
                self._q = self._equatorial2quat()
            elif self._T is not None:
                self._q = self._transform2quat()
        return self._q.reshape(self.shape + (4,))

    @q.setter
    def q(self, q):
        """
        Set the value of the 4 element quaternion vector
        May be 4 element list or array or N x 4 element array,
        where N can be an arbitrary shape. E.g.: (3,2,4) is allowed.

        :param q: list or array of normalized quaternion elements
        """
        q = np.array(q)
        shape = q.shape[:-1]  # Capture input shape sans 4-vector dimension
        q = np.atleast_2d(q)
        if np.any((np.sum(q ** 2, axis=-1, keepdims=True) - 1.0) > 1e-6):
            raise ValueError(
                'Quaternions must be normalized so sum(q**2) == 1; use Quaternion.normalize')

        self._q = q
        flip_q = q[..., 3] < 0
        self._q[flip_q] = -1 * q[flip_q]
        # Erase internal values of other representations
        self._equatorial = None
        self._T = None
        self._shape = shape

    def __repr__(self):
        q = self.q
        if q.ndim == 1:
            return ('<{} q1={:.8f} q2={:.8f} q3={:.8f} q4={:.8f}>'
                    .format(self.__class__.__name__, q[0], q[1], q[2], q[3]))
        return '{}({})'.format(self.__class__.__name__, repr(q))

    @property
    def equatorial(self):
        """Retrieve [RA, Dec, Roll]

        :rtype: numpy array
        """
        if self._equatorial is None:
            if self._q is not None:
                self._equatorial = self._quat2equatorial()
            elif self._T is not None:
                self._q = self._transform2quat()
                self._equatorial = self._quat2equatorial()
        return self._equatorial.reshape(self.shape + (3,))

    @equatorial.setter
    def equatorial(self, equatorial):
        """Set the value of the 3 element equatorial coordinate list [RA,Dec,Roll]
           expects values in degrees
           bounds are not checked

           :param equatorial: list or array [ RA, Dec, Roll] in degrees

        """
        equatorial = np.array(equatorial)
        shape = equatorial.shape[:-1]  # Capture input shape sans 3-vector dimension
        self._equatorial = np.atleast_2d(equatorial)
        self._q = None
        self._transform = None
        self._shape = shape

    @property
    def ra(self):
        """Retrieve RA term from equatorial system in degrees"""
        return self.equatorial[..., 0].reshape(self.shape)[()]

    @property
    def dec(self):
        """Retrieve Dec term from equatorial system in degrees"""
        return self.equatorial[..., 1].reshape(self.shape)[()]

    @property
    def roll(self):
        """Retrieve Roll term from equatorial system in degrees"""
        return self.equatorial[..., 2].reshape(self.shape)[()]

    @staticmethod
    def _get_zero(val):
        """
        Return a version of val that is between -180 <= val < 180
        """
        shape = np.array(val).shape
        val = np.atleast_1d(val)
        val = val % 360
        val[val >= 180] -= 360
        return val.reshape(shape)[()]

    @property
    def ra0(self):
        """
        Return quaternion RA in the range -180 <= roll < 180.
        """
        if self._ra0 is None:
            self._ra0 = self._get_zero(self.ra)
        return self._ra0

    @property
    def roll0(self):
        """
        Return quaternion roll in the range -180 <= roll < 180.
        """
        if self._roll0 is None:
            self._roll0 = self._get_zero(self.roll)
        return self._roll0

    @property
    def pitch(self):
        """
        Return quaternion pitch (same as -dec)
        """
        return -self.dec

    @property
    def yaw(self):
        """
        Return quaternion yaw (same as ra0)
        """
        return self.ra0

    @property
    def transform(self):
        """
        Retrieve the value of the 3x3 rotation/transform matrix

        :returns: 3x3 rotation/transform matrix
        :rtype: numpy array

        """
        if self._T is None:
            if self._q is not None:
                self._T = self._quat2transform()
            elif self._equatorial is not None:
                self._T = self._equatorial2transform()
        return self._T.reshape(self.shape + (3, 3))

    @transform.setter
    def transform(self, t):
        """
        Set the value of the 3x3 rotation/transform matrix

        :param t: 3x3 array/numpy array
        """
        transform = np.array(t)
        shape = transform.shape[:-2]  # Capture input shape sans matrix dimensions
        self._T = np.atleast_3d(transform)
        self._q = None
        self._equatorial = None
        self._shape = shape

    def _apply(self, method, *args, **kwargs):
        """Create a new Quat object, possibly applying a method to self.q.

        Parameters
        ----------
        method : str or callable
            If string, can be 'replicate'  or the name of a relevant
            `~numpy.ndarray` method. In the former case, a new Quat instance
            with unchanged internal data is created, while in the latter the
            method is applied to self.q.
            If a callable, it is directly applied to the above arrays.
            Examples: 'copy', '__getitem__', 'reshape', `~numpy.broadcast_to`.
        args : tuple
            Any positional arguments for ``method``.
        kwargs : dict
            Any keyword arguments for ``method``.


        Examples
        --------
        ::

            copy : ``_apply('copy')``
            reshape : ``_apply('reshape', new_shape)``
        """
        if callable(method):
            apply_method = lambda array: method(array, *args, **kwargs)  # noqa
        else:
            apply_method = operator.methodcaller(method, *args, **kwargs)

        # Use a trick to temporarily make q into a stuctured array with 4
        # floats so that the shape applies to the 4-float item rather than
        # the full array (essentially removing the last dimension).
        qsa = self.q.view(np.dtype([('quat', '4f8')]))
        # view() always leaves a single dimension at end so squeeze it.
        qsa = qsa.squeeze(axis=-1)
        q = apply_method(qsa)['quat']

        # Get a new instance of our class and set its attributes directly.
        out = self.__class__.__new__(self.__class__)
        out._q = np.atleast_2d(q)
        out._shape = q.shape[:-1]

        return out

    def __getitem__(self, item):
        if isinstance(item, tuple) and len(item) > self.ndim:
            raise IndexError('too many indices for array')

        out = self.__class__.__new__(self.__class__)
        q = self.q[item]
        out._q = np.atleast_2d(q)
        out._shape = q.shape[:-1]

        return out

    def _quat2equatorial(self):
        """
        Determine Right Ascension, Declination, and Roll for the quaternion

        :returns: N x (RA, Dec, Roll)
        :rtype: numpy array [ra,dec,roll]
        """

        q = np.atleast_2d(self.q)
        q2 = q ** 2

        # calculate direction cosine matrix elements from $quaternions
        xa = q2[..., 0] - q2[..., 1] - q2[..., 2] + q2[..., 3]
        xb = 2 * (q[..., 0] * q[..., 1] + q[..., 2] * q[..., 3])
        xn = 2 * (q[..., 0] * q[..., 2] - q[..., 1] * q[..., 3])
        yn = 2 * (q[..., 1] * q[..., 2] + q[..., 0] * q[..., 3])
        zn = q2[..., 3] + q2[..., 2] - q2[..., 0] - q2[..., 1]

        # Due to numerical precision this can go negative.  Allow *slightly* negative
        # values but raise an exception otherwise.
        one_minus_xn2 = 1 - xn**2
        if np.any(one_minus_xn2 < 0):
            if np.any(one_minus_xn2 < -1e-12):
                raise ValueError('Unexpected negative norm: {}'.format(one_minus_xn2))
            one_minus_xn2[one_minus_xn2 < 0] = 0

        # ; calculate RA, Dec, Roll from cosine matrix elements
        ra = np.degrees(np.arctan2(xb, xa))
        dec = np.degrees(np.arctan2(xn, np.sqrt(one_minus_xn2)))
        roll = np.degrees(np.arctan2(yn, zn))
        # all negative angles are incremented by 360,
        # the output is in the (0,360) interval instead of in (-180, 180)
        ra[ra < 0] = ra[ra < 0] + 360
        roll[roll < 0] = roll[roll < 0] + 360
        # moveaxis in the following line is a "transpose"
        # from shape (3, N) to (N, 3), where N can be an arbitrary tuple
        # e.g. (3, 2, 5) -> (2, 5, 3) (np.transpose would give (2, 3, 5))
        return np.moveaxis(np.array([ra, dec, roll]), 0, -1)

#  _quat2transform is largely from Enthought's quaternion.rotmat, though this math is
#  probably from Hamilton.
#  License included for completeness
#
# This software is OSI Certified Open Source Software.
# OSI Certified is a certification mark of the Open Source Initiative.
#
# Copyright (c) 2006, Enthought, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  * Neither the name of Enthought, Inc. nor the names of its contributors may
#    be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    def _quat2transform(self):
        """
        Transform a unit quaternion into its corresponding rotation/transform matrix.

        :returns: Nx3x3 transform matrix
        :rtype: numpy array

        """
        q = np.atleast_2d(self.q)

        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        xx2 = x * x * 2.
        yy2 = y * y * 2.
        zz2 = z * z * 2.
        xy2 = x * y * 2.
        wz2 = w * z * 2.
        zx2 = z * x * 2.
        wy2 = w * y * 2.
        yz2 = y * z * 2.
        wx2 = w * x * 2.

        t = np.empty(tuple(q.shape[:-1] + (3, 3)), np.float64)
        t[..., 0, 0] = 1. - yy2 - zz2
        t[..., 0, 1] = xy2 - wz2
        t[..., 0, 2] = zx2 + wy2
        t[..., 1, 0] = xy2 + wz2
        t[..., 1, 1] = 1. - xx2 - zz2
        t[..., 1, 2] = yz2 - wx2
        t[..., 2, 0] = zx2 - wy2
        t[..., 2, 1] = yz2 + wx2
        t[..., 2, 2] = 1. - xx2 - yy2

        return t

    def _equatorial2quat(self):
        """Return quaternion.

        :returns: quaternion
        :rtype: Quat

        """
        return self._transform2quat()

    def _equatorial2transform(self):
        """Construct the transform/rotation matrix from RA,Dec,Roll

        :returns: transform matrix
        :rtype: Nx3x3 numpy array

        """
        ra = np.radians(self.ra)
        dec = np.radians(self.dec)
        roll = np.radians(self.roll)
        ca = np.cos(ra)
        sa = np.sin(ra)
        cd = np.cos(dec)
        sd = np.sin(dec)
        cr = np.cos(roll)
        sr = np.sin(roll)

        # This is the transpose of the transformation matrix (related to translation
        # of original perl code
        rmat = np.array([[ca * cd,                    sa * cd,                  sd],  # noqa
                         [-ca * sd * sr - sa * cr,   -sa * sd * sr + ca * cr,   cd * sr],  # noqa
                         [-ca * sd * cr + sa * sr,   -sa * sd * cr - ca * sr,   cd * cr]])  # noqa

        return np.moveaxis(np.moveaxis(rmat, 0, -1), 0, -2)

    def _transform2quat(self):
        """Construct quaternions from the transform/rotation matrices

        :returns: quaternions formed from transform matrices
        :rtype: numpy array
        """

        T = self.transform
        if T.ndim == 2:
            T = T[np.newaxis]
        den = np.array([1.0 + T[..., 0, 0] - T[..., 1, 1] - T[..., 2, 2],
                        1.0 - T[..., 0, 0] + T[..., 1, 1] - T[..., 2, 2],
                        1.0 - T[..., 0, 0] - T[..., 1, 1] + T[..., 2, 2],
                        1.0 + T[..., 0, 0] + T[..., 1, 1] + T[..., 2, 2]])

        half_rt_q_max = 0.5 * np.sqrt(np.max(den, axis=0))
        max_idx = np.argmax(den, axis=0)
        poss_quat = np.zeros(tuple((4,) + T.shape[:-2] + (4,)))
        denom = 4.0 * half_rt_q_max
        poss_quat[0] = np.moveaxis(
            np.array(
                [half_rt_q_max,
                 (T[..., 1, 0] + T[..., 0, 1]) / denom,
                 (T[..., 2, 0] + T[..., 0, 2]) / denom,
                 (T[..., 2, 1] - T[..., 1, 2]) / denom]), 0, -1)
        poss_quat[1] = np.moveaxis(
            np.array(
                [(T[..., 1, 0] + T[..., 0, 1]) / denom,
                 half_rt_q_max,
                 (T[..., 2, 1] + T[..., 1, 2]) / denom,
                 (T[..., 0, 2] - T[..., 2, 0]) / denom]), 0, -1)
        poss_quat[2] = np.moveaxis(
            np.array(
                [(T[..., 2, 0] + T[..., 0, 2]) / denom,
                 (T[..., 2, 1] + T[..., 1, 2]) / denom,
                 half_rt_q_max,
                 (T[..., 1, 0] - T[..., 0, 1]) / denom]), 0, -1)
        poss_quat[3] = np.moveaxis(
            np.array(
                [(T[..., 2, 1] - T[..., 1, 2]) / denom,
                 (T[..., 0, 2] - T[..., 2, 0]) / denom,
                 (T[..., 1, 0] - T[..., 0, 1]) / denom,
                 half_rt_q_max]), 0, -1)

        q = np.zeros(tuple(T.shape[:-2] + (4,)))
        for idx in range(0, 4):
            max_match = max_idx == idx
            q[max_match] = poss_quat[idx][max_match]

        return q

    def __div__(self, quat2):
        """
        Divide one quaternion by another (or divide N quats by N quats).

        Example usage::

         >>> q1 = Quat((20,30,40))
         >>> q2 = Quat((30,40,50))
         >>> q = q1 / q2

        Performs the operation as q1 * inverse(q2) which is equivalent to
        the inverse(q2) transform followed by the q1 transform.  See the __mul__
        operator help for more explanation on composing quaternions.

        :returns: product q1 * inverse q2
        :rtype: Quat

        """
        return self * quat2.inv()

    __truediv__ = __div__

    def __mul__(self, quat2):
        """
        Multiply quaternion by another (or multiply N quats by N quats).

        Quaternion composition as a multiplication q = q1 * q2 is equivalent to
        applying the q2 transform followed by the q1 transform.  Another way to
        express this is::

          q = Quat(q1.transform @ q2.transform)

        Example usage::

          >>> q1 = Quat((20,30,0))
          >>> q2 = Quat((0,0,40))
          >>> (q1 * q2).equatorial
          array([20., 30., 40.])

        This example first rolls about X by 40 degrees, then rotates that rolled frame
        to RA=20 and Dec=30.  Doing the composition in the other order does a roll about
        (the original) X-axis of the (RA, Dec) = (20, 30) frame, yielding a non-intuitive
        though correct result::

          >>> (q2 * q1).equatorial
          array([ 353.37684725,   34.98868888,   47.499696  ])

        :returns: product q1 * q2
        :rtype: Quat

        """
        q1 = np.atleast_2d(self.q)
        q2 = np.atleast_2d(quat2.q)
        if q1.shape != q2.shape:
            q1, q2 = np.broadcast_arrays(q1, q2)
        mult = np.zeros(q1.shape)
        mult[...,0] =  q1[...,3] * q2[...,0] - q1[...,2] * q2[...,1] + q1[...,1] * q2[...,2] + q1[...,0] * q2[...,3]  # noqa
        mult[...,1] =  q1[...,2] * q2[...,0] + q1[...,3] * q2[...,1] - q1[...,0] * q2[...,2] + q1[...,1] * q2[...,3]  # noqa
        mult[...,2] = -q1[...,1] * q2[...,0] + q1[...,0] * q2[...,1] + q1[...,3] * q2[...,2] + q1[...,2] * q2[...,3]  # noqa
        mult[...,3] = -q1[...,0] * q2[...,0] - q1[...,1] * q2[...,1] - q1[...,2] * q2[...,2] + q1[...,3] * q2[...,3]  # noqa
        shape = self.q.shape if len(self.q.shape) > len(quat2.q.shape) else quat2.q.shape
        return Quat(q=mult.reshape(shape))

    def inv(self):
        """
        Invert the quaternion.

        :returns: inverted quaternion
        :rtype: Quat
        """
        q = np.array(self.q)
        q[..., 3] *= -1
        return Quat(q=q)


    def dq(self, q2=None, **kwargs):
        """
        Return a delta quaternion ``dq`` such that ``q2 = self * dq``.
        I works with any argument that instantiates a ``Quat`` object.

        This method returns the delta quaternion which represents the transformation
        from the frame of this quaternion (``self``) to ``q2``.

          q = Quat(q1.transform @ q2.transform)

        Example usage::

          >>> q1 = Quat((20, 30, 0))
          >>> q2 = Quat((20, 30.1, 1))
          >>> dq = q1.dq(q2)
          >>> dq.equatorial
          array([  1.79974166e-15,   1.00000000e-01,   1.00000000e+00])

        :param: q2 Quat or array

        ``q2`` must have the same shape as self.

        :returns: Quat
        :rtype: numpy array

        """
        if q2 is None or not isinstance(q2, Quat):
            q2 = Quat(q2, **kwargs)
        return self.inv() * q2


    def __setstate__(self, state):
        # this method is called by the pickle module when unpickling an instance of this class.
        # It receives the dictionary of the unpickled state and must update the __dict__ of this
        # instance. This is the place where we "upgrade" pickled instances of earlier versions of
        # the class.
        if '_shape' not in state:
            # if _shape is not there, then this is a non-vectorized Quat (versions before 3.5.0).
            # Non-vectorized quaternions were basically scalars, with shape ().
            state['_shape'] = ()

        self.__dict__.update(state)

    @classmethod
    def rotate_x_to_vec(cls, vec, method='radec'):
        """Generate quaternion that rotates X-axis into ``vec``.

        The ``method`` parameter can take one of three values: "shortest",
        "keep_z", or "radec" (default).  The "shortest" method takes the shortest
        path between the two vectors.  The "radec" method does the transformation
        as the corresponding (RA, Dec, Roll=0) attitude.  The "keep_z" method does
        a roll about X-axis (followed by the "shortest" path) such that the
        transformed Z-axis is in the original X-Z plane.  In equations::

        T: "shortest" quaternion taking X-axis to vec
        Rx(theta): Rotation by theta about X-axis = [[1,0,0], [0,c,s], [0,-s,c]]
        Z: Z-axis [0,0,1]

        [T * Rx(theta) * Z]_y = 0
        T[1,1] * sin(theta) + T[1,2]*cos(theta) = 0
        theta = atan2(T[1,2], T[1,1])

        :param vec: Input 3-vector
        :param method: method for determining path (shortest|keep_z|radec)
        :returns: Quaternion object
        """
        x = np.array([1., 0, 0])
        vec = normalize(vec)
        if vec.shape != (3,):
            raise ValueError('vec must be a single 3-vector')

        if method in ("shortest", "keep_z"):
            dot = np.dot(x, vec)
            if abs(dot) > 1 - 1e-8:
                x = normalize(np.array([1., 0., 1e-7]))
                dot = np.dot(vec, x)
            angle = np.arccos(dot)
            axis = normalize(np.cross(x, vec))
            sin_a = np.sin(angle / 2)
            cos_a = np.cos(angle / 2)
            q = cls([axis[0] * sin_a,
                     axis[1] * sin_a,
                     axis[2] * sin_a,
                     cos_a])

            if method == "keep_z":
                T = q.transform
                theta = np.arctan2(T[1, 2], T[1, 1])
                qroll = cls([0, 0, np.rad2deg(theta)])
                q = q * qroll

        elif method == "radec":
            ra = np.degrees(np.arctan2(vec[1], vec[0]))
            dec = np.degrees(np.arcsin(vec[2]))
            q = cls([ra, dec, 0])

        else:
            raise ValueError('method must be one of "shortest", "keep_z" or "radec"')

        return q


    def rotate_about_vec(self, vec, alpha):
        """Rotate self about a 3-vector

        :param vec: 3-element array-like
            Single vector to rotate about
        :param alpha: float
            Angle to rotate by in degrees

        :returns: Quat
            Rotated attitude
        """
        # TODO: fix these limitations on shapes, starting with quat multiplication
        if self.shape != ():
            raise ValueError('quaternion must be a scalar')

        vec = np.asarray(vec)
        if vec.shape != (3,):
            raise ValueError('vec must be a single 3-vector')
        vec = vec / np.linalg.norm(vec)

        alpha = np.asarray(alpha)
        if alpha.shape != ():
            raise ValueError('alpha must be a scalar')

        alpha = np.deg2rad(alpha)
        sin_alpha = np.sin(alpha / 2.0)
        cos_alpha = np.cos(alpha / 2.0)

        dq = Quat([sin_alpha * vec[0], sin_alpha * vec[1], sin_alpha * vec[2], cos_alpha])
        att_out = dq * self
        return att_out


    @classmethod
    def from_attitude(cls, att):
        """Initial Quat from attitude(s)

        This allows initialization from several possibilities that are less
        strict than the normal initializer:
        - Quat already (returns same)
        - Input that works with Quat already
        - Input that initializes a float ndarray which has shape[-1] == 3 or 4
        - Input that initializes an object ndarray where each element inits Quat,
        e.g. a list of Quat

        Parameters
        ----------
        att : Quaternion-like
            Attitude(s)

        Returns
        -------
        Quat
            Attitude(s) as a Quat
        """
        if isinstance(att, Quat):
            return att.copy()

        # Input that works with Quat already
        try:
            att = cls(att)
        except Exception:
            pass
        else:
            return att

        # Input that initializes a float ndarray which has shape[-1] == 3 or 4
        try:
            att_array = np.asarray(att, dtype=np.float64)
        except Exception:
            pass
        else:
            if att_array.shape[-1] == 4:
                return cls(q=att_array)
            elif att_array.shape[-1] == 3:
                return cls(equatorial=att_array)
            else:
                raise ValueError("Float input must be a Nx3 or Nx4 array")

        # Input that initializes an object ndarray where each element inits Quat,
        # e.g. a list of Quat
        try:
            att_array = np.asarray(att, dtype=object)
            qs = []
            for val in att_array.ravel():
                qs.append(val.q if isinstance(val, Quat) else cls(val).q)
            qs = np.array(qs, dtype=np.float64).reshape(att_array.shape + (4,))
            return cls(q=qs)
        except Exception:
            pass

        raise ValueError(f"Unable to initialize {cls.__name__} from {att!r}")



def normalize(array):
    """
    Normalize a 4 (or Nx4) element array/list/numpy.array for use as a quaternion

    :param array: 4 or Nx4 element list/array
    :returns: normalized array
    :rtype: numpy array

    """
    quat = np.array(array)
    return np.squeeze(quat / np.sqrt(np.sum(quat * quat, axis=-1, keepdims=True)))
