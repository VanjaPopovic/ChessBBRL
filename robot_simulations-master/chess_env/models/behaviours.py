import torch
import torch.nn as nn


class Approach(nn.Module):

    """
        Neural network that implements the apporoach behaviour
    """

    def __init__(self, input_size, log_sig_max, log_sig_min, weights_init_fn):
        super(Approach, self).__init__()

        self.log_sig_max = log_sig_max
        self.log_sig_min = log_sig_min

        self.approach_x_mean = nn.Linear(input_size, 1)
        self.approach_x_log_std = nn.Linear(input_size, 1)
        self.approach_y_mean = nn.Linear(input_size, 1)
        self.approach_y_log_std = nn.Linear(input_size, 1)
        self.approach_z_mean = nn.Linear(input_size, 1)
        self.approach_z_log_std = nn.Linear(input_size, 1)

        self.apply(weights_init_fn)

    def forward(self, x):

        mean_x = self.approach_x_mean(x)
        log_std_x = self.approach_x_log_std(x)
        log_std_x = torch.clamp(
            log_std_x, min=self.log_sig_min, max=self.log_sig_max)

        mean_y = self.approach_y_mean(x)
        log_std_y = self.approach_y_log_std(x)
        log_std_y = torch.clamp(
            log_std_y, min=self.log_sig_min, max=self.log_sig_max)

        mean_z = self.approach_z_mean(x)
        log_std_z = self.approach_z_log_std(x)
        log_std_z = torch.clamp(
            log_std_z, min=self.log_sig_min, max=self.log_sig_max)

        means_x = torch.cat([mean_x, mean_y, mean_z])
        log_stds_x = torch.cat([log_std_x, log_std_y, log_std_z])

        return means_x, log_stds_x.exp()


class Grasp(nn.Module):

    """
        Neural network that implements the grasp behaviour
    """

    def __init__(self, input_size, log_sig_max, log_sig_min, weights_init_fn):
        super(Grasp, self).__init__()

        self.log_sig_max = log_sig_max
        self.log_sig_min = log_sig_min

        self.grasp_x_mean = nn.Linear(input_size, 1)
        self.grasp_x_log_std = nn.Linear(input_size, 1)
        self.grasp_y_mean = nn.Linear(input_size, 1)
        self.grasp_y_log_std = nn.Linear(input_size, 1)
        self.grasp_z_mean = nn.Linear(input_size, 1)
        self.grasp_z_log_std = nn.Linear(input_size, 1)
        self.grap_r_mean = nn.Linear(input_size, 1)
        self.grap_r_log_std = nn.Linear(input_size, 1)

        self.apply(weights_init_fn)

    def forward(self, x):

        mean_x = self.grasp_x_mean(x)
        log_std_x = self.grasp_x_log_std(x)
        log_std_x = torch.clamp(
            log_std_x, min=self.log_sig_min, max=self.log_sig_max)

        mean_y = self.grasp_y_mean(x)
        log_std_y = self.grasp_y_log_std(x)
        log_std_y = torch.clamp(
            log_std_y, min=self.log_sig_min, max=self.log_sig_max)

        mean_z = self.grasp_z_mean(x)
        log_std_z = self.grasp_z_log_std(x)
        log_std_z = torch.clamp(
            log_std_z, min=self.log_sig_min, max=self.log_sig_max)

        # mean_r = self.grap_r_mean(x)
        # log_std_r = self.grap_r_log_std(x)
        # log_std_r = torch.clamp(log_std_r, min=self.log_sig_min, max=self.log_sig_max)

        means_x = torch.cat([mean_x, mean_y, mean_z])  # , mean_r])
        # , log_std_r])
        log_stds_x = torch.cat([log_std_x, log_std_y, log_std_z])

        return means_x, log_stds_x.exp()


class Retract(nn.Module):

    """
        Neural network that implements the retract behaviour
    """

    def __init__(self, input_size, log_sig_max, log_sig_min, weights_init_fn):
        super(Retract, self).__init__()

        self.log_sig_max = log_sig_max
        self.log_sig_min = log_sig_min

        self.retract_x_mean = nn.Linear(input_size, 1)
        self.retract_x_log_std = nn.Linear(input_size, 1)
        self.retract_y_mean = nn.Linear(input_size, 1)
        self.retract_y_log_std = nn.Linear(input_size, 1)
        self.retract_z_mean = nn.Linear(input_size, 1)
        self.retract_z_log_std = nn.Linear(input_size, 1)

        self.apply(weights_init_fn)

    def forward(self, x):

        mean_x = self.retract_x_mean(x)
        log_std_x = self.retract_x_log_std(x)
        log_std_x = torch.clamp(
            log_std_x, min=self.log_sig_min, max=self.log_sig_max)

        mean_y = self.retract_y_mean(x)
        log_std_y = self.retract_y_log_std(x)
        log_std_y = torch.clamp(
            log_std_y, min=self.log_sig_min, max=self.log_sig_max)

        mean_z = self.retract_z_mean(x)
        log_std_z = self.retract_z_log_std(x)
        log_std_z = torch.clamp(
            log_std_z, min=self.log_sig_min, max=self.log_sig_max)

        means_x = torch.cat([mean_x, mean_y, mean_z])
        log_stds_x = torch.cat([log_std_x, log_std_y, log_std_z])

        return means_x, log_stds_x.exp()


class Place(nn.Module):

    """
        Neural network that implements the place behaviour
    """

    def __init__(self, input_size, log_sig_max, log_sig_min, weights_init_fn):
        super(Place, self).__init__()

        self.log_sig_max = log_sig_max
        self.log_sig_min = log_sig_min

        self.place_x_mean = nn.Linear(input_size, 1)
        self.place_x_log_std = nn.Linear(input_size, 1)
        self.place_y_mean = nn.Linear(input_size, 1)
        self.place_y_log_std = nn.Linear(input_size, 1)
        self.place_z_mean = nn.Linear(input_size, 1)
        self.place_z_log_std = nn.Linear(input_size, 1)
        self.place_r_mean = nn.Linear(input_size, 1)
        self.place_r_log_std = nn.Linear(input_size, 1)

        self.apply(weights_init_fn)

    def forward(self, x):

        mean_x = self.place_x_mean(x)
        log_std_x = self.place_x_log_std(x)
        log_std_x = torch.clamp(
            log_std_x, min=self.log_sig_min, max=self.log_sig_max)

        mean_y = self.place_y_mean(x)
        log_std_y = self.place_y_log_std(x)
        log_std_y = torch.clamp(
            log_std_y, min=self.log_sig_min, max=self.log_sig_max)

        mean_z = self.place_z_mean(x)
        log_std_z = self.place_z_log_std(x)
        log_std_z = torch.clamp(
            log_std_z, min=self.log_sig_min, max=self.log_sig_max)

        # mean_r = self.grap_r_mean(x)
        # log_std_r = self.grap_r_log_std(x)
        # log_std_r = torch.clamp(log_std_r, min=self.log_sig_min, max=self.log_sig_max)

        means_x = torch.cat([mean_x, mean_y, mean_z])  # , mean_r])
        # , log_std_r])
        log_stds_x = torch.cat([log_std_x, log_std_y, log_std_z])

        return means_x, log_stds_x.exp()
