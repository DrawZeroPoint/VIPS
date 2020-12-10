import datetime

import numpy as np

from experiments.VIPS.abstract_vips_experiment import AbstractVIPSExperiment
from experiments.lnpdfs.create_target_lnpfs import build_target_likelihood_panda_arm_waiter, panda_fk
from robotics.kinematics import fk_in_space
from plotting.default_plot import default_plots
import matplotlib.pyplot as plt
from plotting.visualize_n_link import visualize_mixture
from experiments.VIPS.configs.ConfigUtils import ConfigUtils


class PandaArm(AbstractVIPSExperiment):
    def __init__(self, num_dimensions, num_initial_components,
                 initial_mixture_prior_variance, cart_likelihood_var, conf_likelihood_var, config):
        self.cart_likelihood_var = cart_likelihood_var
        self.conf_likelihood_var = conf_likelihood_var
        filepath = "panda_arm/" + str(num_dimensions) + '/' + datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")+'/'
        AbstractVIPSExperiment.__init__(self, filepath, num_dimensions, num_initial_components, initial_mixture_prior_variance)
        [self.target_lnpdf, self.prior, self.prior_chol] = build_target_likelihood_panda_arm_waiter(
            num_dimensions, conf_likelihood_var, cart_likelihood_var
        )
        self.groundtruth_samples = None
        self.groundtruth_lnpdfs = None
        self.config = config

        # add plot for robot visualization
        def all_plots(sampler):
            plt.figure(100)
            plt.clf()
            plt.title("Robot visualization")
            [weights, means, _] = sampler.vips_c.get_model()
            max_id = weights.argmax()
            print(max_id, weights[max_id])
            print(np.dot(means[max_id], 180. / np.pi))
            x, y, z, qx, qy, qz, qw = panda_fk(means[max_id])
            print('x: ', x)
            print('y: ', y)
            print('z: ', z)
            print('qx: ', qx)
            print('qy: ', qy)
            print('qz: ', qz)
            print('qw: ', qw)

        changes_in_functions = {
            'plot_fctn': all_plots,
        }
        ConfigUtils.merge_FUNCTIONS(config, changes_in_functions)

    def obtain_groundtruth(self):
        self.groundtruth_samples = np.load(self.data_path + "groundtruth/panda_arm/samples_w.npy")
        self.groundtruth_lnpdfs = self.target_lnpdf(self.groundtruth_samples)

    def run_experiment(self):
        self.run_experiment_VIPS(self.target_lnpdf, self.config, self.groundtruth_samples, self.groundtruth_lnpdfs)


if __name__ == '__main__':
    import experiments.VIPS.configs.fast_adding_old_reusage as config

    config.COMMON['mmd_alpha'] = 6
    num_dimensions = 7
    cart_likelihood_var = np.array([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4])

    conf_likelihood_var = np.ones(num_dimensions)
    use_known_prior = True
    if use_known_prior:
        initial_prior = conf_likelihood_var.copy()
    else:
        initial_prior = np.ones(num_dimensions)

    experiment = PandaArm(num_dimensions=num_dimensions,
                          num_initial_components=50,
                          initial_mixture_prior_variance=initial_prior,
                          cart_likelihood_var=cart_likelihood_var,
                          conf_likelihood_var=conf_likelihood_var,
                          config=config)
    experiment.obtain_groundtruth()
    experiment.enable_progress_logging(config, 5)
    experiment.run_experiment()
