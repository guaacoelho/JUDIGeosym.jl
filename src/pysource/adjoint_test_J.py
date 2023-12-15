import numpy as np
from argparse import ArgumentParser
from scipy import ndimage
from devito import inner

from sources import RickerSource, Receiver
from models import Model

from propagators import forward, born, gradient


parser = ArgumentParser(description="Adjoint test args")
parser.add_argument("--tti", default=False, action='store_true',
                    help="Test acoustic or tti")
parser.add_argument("--viscoacoustic", default=False, action='store_true',
                    help="Test viscoacoustic")
parser.add_argument("--fs", default=False, action='store_true',
                    help="Test with free surface")
parser.add_argument('-so', dest='space_order', default=8, type=int,
                    help="Spatial discretization order")
parser.add_argument('-nlayer', dest='nlayer', default=3, type=int,
                    help="Number of layers in model")

args = parser.parse_args()
is_tti = args.tti
is_viscoacoustic = args.viscoacoustic
so = args.space_order

dtype = np.float32

# Model
shape = (301, 301)
spacing = (10., 10.)
origin = (0., 0.)
m = np.empty(shape, dtype=dtype)
rho = np.empty(shape, dtype=dtype)
m[:] = 1/1.5**2  # Top velocity (background)
rho[:] = 1.0
m_i = np.linspace(1/1.5**2, 1/4.5**2, args.nlayer)
for i in range(1, args.nlayer):
    m[..., i*int(shape[-1] / args.nlayer):] = m_i[i]  # Bottom velocity

m0 = ndimage.gaussian_filter(m, sigma=10)
m0[m > 1/1.51**2] = m[m > 1/1.51**2]
m0 = ndimage.gaussian_filter(m0, sigma=3)
rho0 = (m0**(-.5)+.5)/2
dm = m0 - m
dm[:, -1] = 0.
# Set up model structures
v0 = m0**(-.5)
if is_tti:
    model = Model(shape=shape, origin=origin, spacing=spacing, dtype=dtype,
                  m=m0, epsilon=.045*(v0-1.5), delta=.03*(v0-1.5),
                  fs=args.fs, rho=rho0, theta=.1*(v0-1.5), dm=dm, space_order=so)
elif is_viscoacoustic:
    qp0 = np.empty(shape, dtype=dtype)
    qp0[:] = 3.516*((v0[:]*1000.)**2.2)*10**(-6)
    model = Model(shape=shape, origin=origin, spacing=spacing, dtype=dtype,
                  fs=args.fs, m=m0, rho=rho0, qp=qp0, dm=dm, space_order=so)
else:
    model = Model(shape=shape, origin=origin, spacing=spacing, dtype=dtype,
                  fs=args.fs, m=m0, rho=rho0, dm=dm, space_order=so)


# Precisei definir esses valores para passar para o model e definir que é o acústico.
# Se eu definisse valores constantes não seria definido como função e sim como constante, 
# por isso a definição dos valores 0.5.
lam = np.ones(shape, dtype=dtype)
lam[6:] = 0.5
mu = np.ones(shape, dtype=dtype)
mu[6:] = 0.5
model_elas = Model(shape=shape, origin=origin, spacing=spacing, dtype=dtype,
                fs=args.fs, rho=rho0, dm=dm, mu=mu, lam=lam, space_order=so)

# Time axis
t0 = 0.
tn = 2000.
dt = model.critical_dt
nt = int(1 + (tn-t0) / dt)
time_axis = np.linspace(t0, tn, nt)

# Source
f1 = 0.0125
src = RickerSource(name='src', grid=model.grid, f0=f1, time=time_axis)
src.coordinates.data[0, :] = np.array(model.domain_size) * 0.5
src.coordinates.data[0, -1] = 20.
weights = np.zeros(model.grid.shape)
weights[141:161, 141:161] = np.ones((20, 20))

# Receiver for observed data
rec_t = Receiver(name='rec_t', grid=model.grid, npoint=301, ntime=nt)
rec_t.coordinates.data[:, 0] = np.linspace(0., 3000., num=301)
rec_t.coordinates.data[:, 1] = 20.

rec_vx = Receiver(name='rec_vx', grid=model.grid, npoint=301, ntime=nt)
rec_vx.coordinates.data[:, 0] = np.linspace(0., 3000., num=301)
rec_vx.coordinates.data[:, 1] = 20.

rec_vz = Receiver(name='rec_vz', grid=model.grid, npoint=301, ntime=nt)
rec_vz.coordinates.data[:, 0] = np.linspace(0., 3000., num=301)
rec_vz.coordinates.data[:, 1] = 20.

# Linearized data
print("Forward J")
dD_hat, u0l, _, _ = born(model, src.coordinates.data, rec_t.coordinates.data,
                      src.data, save=True, f0=f1)
# dD_hat, u0l, _ = born(model, None, rec_t.coordinates.data,
#                       src.data, save=True, ws=weights)
# Forward
print("Forward")
rout, u0, _, _ = forward(model_elas, src.coordinates.data, rec_t.coordinates.data,
                   src.data, save=True, f0=f1)
# _, u0, _ = forward(model, None, rec_t.coordinates.data,
#                    src.data, save=True, ws=weights)

# gradient
print("Adjoint J")
par="lam-mu"
grad1, grad2, grad3, _, _ = gradient(model_elas, rec_vx, rec_vz, dD_hat.data, rec_t.coordinates.data, u0[0], f0=f1, par=par)


print(np.max(grad1.data))
print(np.max(grad2.data))
print(np.max(grad3.data))

