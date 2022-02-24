import numpy as np
import popkinmocks as pkm

# get ssps and cube
ssps = pkm.model_grids.milesSSPs(lmd_min=4800,
                                 lmd_max=5700,
                                 thin_age=3,
                                 thin_z=2)
ssps.logarithmically_resample(dv=75.)
ssps.calculate_fourier_transform(pad=False)
ssps.get_light_weights()
cube = pkm.ifu_cube.IFUCube(ssps=ssps, nx=25, ny=25, xrng=(-1,1), yrng=(-1,1))
# make component 1
gc1 = pkm.components.growingDisk(cube=cube,
                                 rotation=0.,
                                 center=(0,0))
gc1.set_p_t(lmd=2., phi=0.8)
gc1.set_p_x_t(sig_x_lims=(0.5, 0.2),
              sig_y_lims=(0.03, 0.1),
              alpha_lims=(1.2, 0.8))
gc1.set_t_dep(sig_x=0.5,
              sig_y=0.1,
              alpha=3.,
              t_dep_in=0.5,
              t_dep_out=5.)
gc1.set_p_z_tx()
gc1.set_mu_v(sig_x_lims=(0.5, 0.1),
             sig_y_lims=(0.1, 0.1),
             rmax_lims=(1., 0.045454545454545456),
             vmax_lims=(350., 150.),
             vinf_lims=(50., 10.))
gc1.set_sig_v(sig_x_lims=(0.7, 0.1),
             sig_y_lims=(0.2, 0.1),
             alpha_lims=(3.0, 2.5),
             sig_v_in_lims=(340., 230.),
             sig_v_out_lims=(100., 170))
gc1.evaluate_ybar()
# make component 2
gc2 = pkm.components.growingDisk(cube=cube, rotation=0., center=(0.,0.))
gc2.set_p_t(lmd=7., phi=0.4)
gc2.set_p_x_t(sig_x_lims=(0.05, 0.2),
              sig_y_lims=(0.05, 0.2),
              alpha_lims=(1.5, 0.5))
gc2.set_t_dep(sig_x=0.5,
             sig_y=0.1,
             alpha=3.,
             t_dep_in=0.5,
             t_dep_out=5.0)
gc2.set_p_z_tx()
gc2.set_mu_v(sig_x_lims=(0.5, 0.1),
            sig_y_lims=(0.1, 0.1),
            rmax_lims=(0.5/2., 0.1/1.5),
            vmax_lims=(-100., -100.),
            vinf_lims=(-70., -70))
gc2.set_sig_v(sig_x_lims=(0.7, 0.1),
             sig_y_lims=(0.6, 0.12),
             alpha_lims=(3.0, 2.5),
             sig_v_in_lims=(100., 100.),
             sig_v_out_lims=(30., 30))
gc2.evaluate_ybar()
# make stream component
stream = pkm.components.stream(cube=cube)
stream.set_p_t(lmd=5., phi=0.5)
stream.set_p_x(theta_lims=[-0.25*np.pi, 0.75*np.pi],
               mu_r_lims=[0.9, 0.3],
               sig=0.1,
               nsmp=100)
stream.set_p_z_t(t_dep=9.9)
stream.set_p_v_x(mu_v_lims=[-200,250], sig_v=50.)
stream.evaluate_ybar()
# combine components
cmp_list = [gc1, gc2, stream]
w1, w2, w_stream = 0.7, 0.29, 0.01
weights = [w1, w2, w_stream]
cube.combine_components(component_list=cmp_list, weights=weights)
cube.add_noise(snr=100.)
# save
# save
cube.save_data(direc='../lores/', fname='mockcube.dill')
cube.save_numpy(direc='../lores/', fname='mockcube.npz')
