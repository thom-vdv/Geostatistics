import numpy as np
import math
import matplotlib.pyplot as plt

def distance_latlon(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth's surface.

    Args:
        lat1 (float): Latitude of the first point in degrees.
        lon1 (float): Longitude of the first point in degrees.
        lat2 (float): Latitude of the second point in degrees.
        lon2 (float): Longitude of the second point in degrees.

    Returns:
        float: Distance between the two points in kilometers.
    """
    # Radius of the Earth in kilometers
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Compute differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    # c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    c = 2 * np.arcsin(np.sqrt(a))

    # Distance in kilometers
    distance = R * c
    return distance

def distance_euclidean(x1, y1, x2, y2):
    """
    Calculate the Euclidean distance between two points in a 2D space.

    Args:
        x1 (float): x-coordinate of the first point.
        y1 (float): y-coordinate of the first point.
        x2 (float): x-coordinate of the second point.
        y2 (float): y-coordinate of the second point.

    Returns:
        float: Euclidean distance between the two points.
    """
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def azimuth_latlon(lat1, lon1, lat2, lon2):
    """
    Calculate the azimuth (bearing) between two points on the Earth's surface.

    Args:
        lat1 (float): Latitude of the first point in degrees.
        lon1 (float): Longitude of the first point in degrees.
        lat2 (float): Latitude of the second point in degrees.
        lon2 (float): Longitude of the second point in degrees.

    Returns:
        float: Azimuth (bearing) in degrees from the first point to the second point.
    """
    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Compute the differences
    dlon = lon2_rad - lon1_rad

    # Calculate azimuth
    x = math.sin(dlon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
    azimuth_rad = math.atan2(x, y)

    # Normalize the azimuth to the range [0, 360)
    azimuth = (azimuth_rad + (2*np.pi)) % (2*np.pi)

    return azimuth

def lin_vgm(dist,n,s,r):
    # distance, nugget, sill, range
    gamma = n + (s-n)*(dist/r)
    gamma[np.where(dist<=0)] = 0
    gamma[np.where(dist>=r)] = s
    return gamma

def sph_vgm(dist,n,s,r):
    # distance, nugget, sill, range
    # the quantity s-n is called the partial sill
    gamma = n + (s-n)*(3*dist/(2*r) - dist**3/(2*r**3))
    gamma[np.where(dist<=0)] = 0
    gamma[np.where(dist>=r)] = s
    return gamma

def exp_vgm(dist,n,s,r):
    # dist = distance, n = nugget, s = sill, r = pseudo-range
    # for an exponential model, the pseudo-range is the range at which 95% of the sill is reached.
    gamma = n + (s-n)*(1-np.exp(-3*dist/r))
    gamma[np.where(dist<=0)] = 0
    return gamma

def gauss_vgm(dist,n,s,r):
    # distance, nugget, sill, range
    gamma = n + (s-n)*(1-np.exp(-3*dist**2/r**2))
    gamma[np.where(dist<=0)] = 0
    return gamma


class Variogram:
    def __init__(self):
        self.var_matrix = None
        self.theta_matrix = None
        self.dist_matrix = None
        self.model = None

    def calculate(self, x, y, data, distance_function='euclidean'):
        """
        Calculate the variogram cloud for given data points.
        
        Parameters:
        x (array-like): Array of x coordinates (e.g., longitude).
        y (array-like): Array of y coordinates (e.g., latitude).
        data (array-like): Array of data values at the coordinates.
        distance_function (str): The distance function to use ('euclidean' or 'latlon').
        
        Returns:
        tuple: Variogram cloud, distance list, theta list, and matrices.
        """
        def squared_diff(val1, val2):
            return (val1 - val2)**2

        # Check if input arrays are of the same length
        if len(x) != len(y) or len(x) != len(data):
            raise ValueError("Input arrays x, y, and data must have the same length.")

        if distance_function == 'euclidean':
            dist_func = distance_euclidean
        elif distance_function == 'latlon':
            dist_func = distance_latlon
        else:
            raise ValueError("distance_function must be either 'euclidean' or 'latlon'")

        # Size of the input data
        size = np.size(x)

        #initialize matrices
        dist_matrix = np.zeros((size, size))
        var_matrix = np.zeros((size, size))
        theta_matrix = np.zeros((size, size))

        # loop over every point
        for i, (lat1, lon1) in enumerate(zip(x, y)):
            # loop over upper diagonal of matrix 
            for j, (lat2, lon2) in enumerate(zip(x[i:], y[i:])):
                dist = dist_func(lat1, lon1, lat2, lon2)
                dist_matrix[i, i+j] = dist
                dist_matrix[i+j, i] = dist

                diff = squared_diff(data[i], data[i+j])
                var_matrix[i, i+j] = diff
                var_matrix[i+j, i] = diff

                theta_matrix[i, i+j] = azimuth_latlon(lat1, lon1, lat2, lon2)
                theta_matrix[i+j, i] = (azimuth_latlon(lat1, lon1, lat2, lon2) + np.pi) % (2*np.pi)

        self.var_matrix = var_matrix
        self.theta_matrix = theta_matrix
        self.dist_matrix = dist_matrix

    def add_model(self, model, sill, range, nugget):
        """
        Add a model to the variogram.
        
        Parameters:
        model (str): Type of model ('linear', 'spherical', 'exponential', 'gaussian').
        sill (float): Sill of the model.
        range (float): Range of the model.
        nugget (float): Nugget effect of the model.
        
        Returns:
        None
        """
        if model == 'linear':
            model_func = lin_vgm
        elif model == 'spherical': 
            model_func = sph_vgm
        elif model == 'exponential':
            model_func = exp_vgm
        elif model == 'gaussian':
            model_func = gauss_vgm
        else:
            raise ValueError("Model must be one of ['linear', 'spherical', 'exponential', 'gaussian']")
        
        self.model = model_func
        self.sill = sill
        self.range = range
        self.nugget = nugget

    def plot_isotropic(self, bins=10, cloud=True, ax=None):
        """
        Plot the variogram cloud.
        
        Parameters:
        bins (int): Number of bins for averaging the variogram cloud.
        cloud (bool): If True, plot the variogram cloud; if False, plot the binned averages.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new figure and axes are created.
        """
        # Flatten the matrices and remove zeros
        var_cloud = self.var_matrix[np.triu_indices_from(self.var_matrix, k=1)].flatten()
        dist_list = self.dist_matrix[np.triu_indices_from(self.dist_matrix, k=1)].flatten()

        return self._plot_isotropic(dist_list, var_cloud, bins=bins, cloud=cloud, ax=ax)

    def _plot_isotropic(self, dist_list, var_cloud, bins=10, cloud=True, ax=None):
        """
        Internal method to plot the variogram cloud or binned averages.
        
        Parameters:
        dist_list (array-like): List of distances.
        var_cloud (array-like): Variogram cloud values.
        bins (int): Number of bins for averaging the variogram cloud.
        cloud (bool): If True, plot the variogram cloud; if False, plot the binned averages.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new figure and axes are created.
        """
        
        interval = (np.max(dist_list) - np.min(dist_list)) / bins

        # Initialize bin_list as a list of empty lists
        bin_list = [[] for _ in range(bins)]

        for val, dist in zip(var_cloud, dist_list):
            bin_idx = int(dist // interval)
            # Ensure bin_idx does not exceed bins-1
            bin_idx = min(bin_idx, bins - 1)
            bin_list[bin_idx].append(val)

        bin_centers = np.arange(0, bins) * interval + 0.5 * interval
        bin_averages = np.array([np.mean(bin) if len(bin) > 0 else np.nan for bin in bin_list])

        if ax is None:
            fig, ax = plt.subplots()

        # Plot the variogram cloud
        if cloud:
            ax.scatter(dist_list, var_cloud / 2, s=0.1)
        ax.set_xlabel('Distance')
        ax.set_ylabel('$S_{ij} / 2$')

        # Plot the binned averages
        ax.scatter(bin_centers, bin_averages, color='black', marker='+')

        if self.model is not None:
            # Generate model values for plotting
            model_distances = np.linspace(0, np.max(dist_list), 100)
            model_values = self.model(model_distances, self.nugget, self.sill, self.range)
            ax.plot(model_distances, model_values, color='red')

        return ax
    
    def plot_anisotropic(self, bins_radius=10, bins_heading=36, cloud=True, ax=None):    
        """
        Plot the variogram cloud in polar coordinates, binned by distance and azimuth.
        Parameters:
        bins_radius (int): Number of bins for distance.
        bins_heading (int): Number of bins for azimuth.
        cloud (bool): If True, plot the variogram cloud; if False, plot the binned averages.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new figure and axes are created.
        """

        # Flatten the matrices and remove zeros
        var_cloud = self.var_matrix[np.triu_indices_from(self.var_matrix, k=1)].flatten()
        dist_list = self.dist_matrix[np.triu_indices_from(self.dist_matrix, k=1)].flatten()
        theta_list = self.theta_matrix[np.triu_indices_from(self.theta_matrix, k=1)].flatten()

        # Define bin edges for radius and azimuth
        radius_bins = np.linspace(0, np.max(dist_list), bins_radius + 1)
        azimuth_bins = np.linspace(0, 2 * np.pi, bins_heading + 1)

        # Digitize the data into bins
        radius_indices = np.digitize(dist_list, radius_bins) - 1
        azimuth_indices = np.digitize(theta_list, azimuth_bins) - 1

        if ax is None:
            if cloud:
                fig, ax = plt.subplots(1, 2, subplot_kw={'projection': 'polar'})
                ax[0].set_theta_zero_location('N')
                ax[1].set_theta_zero_location('N')
            else:
                fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'})
                ax = [ax]
                ax[0].set_theta_zero_location('N')

        # Create a 2D histogram
        histogram = np.zeros((len(radius_bins) - 1, len(azimuth_bins) - 1))
        for r_idx, a_idx, value in zip(radius_indices, azimuth_indices, var_cloud):
            if 0 <= r_idx < histogram.shape[0] and 0 <= a_idx < histogram.shape[1]:
                histogram[r_idx, a_idx] += value

        # Normalize the histogram by the number of points in each bin
        counts, _, _ = np.histogram2d(dist_list, theta_list, bins=[radius_bins, azimuth_bins])
        histogram = np.divide(histogram, counts, out=np.zeros_like(histogram), where=counts != 0)

        # Plot the binned polar plot
        r, theta = np.meshgrid(radius_bins, azimuth_bins)
        c = ax[0].pcolormesh(theta, r, histogram.T, cmap="viridis", shading='auto', vmin=0, vmax=0.01)

        if cloud:
            ax[1].scatter(theta_list, dist_list, s=0.1, c=var_cloud, cmap="viridis", vmin=0, vmax=0.01)
            ax[1].grid(True)

        fig.colorbar(c, ax=ax, label="Mean $\gamma(d)$")


    def plot_directional(self, azimuth, tol=5, bins=10, cloud=True, ax=None):
        """
        Plot the variogram cloud for a specific azimuth direction.
        
        Parameters:
        azimuth (float): Azimuth angle in degrees.
        tol (float): Maximum allowed absolute deviation from azimuth.
        bins (int): Number of bins for averaging the variogram cloud.
        cloud (bool): If True, plot the variogram cloud; if False, plot the binned averages.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new figure and axes are created.
        """

        # Convert azimuth to radians
        azimuth_rad = np.radians(azimuth % 360)

        # Flatten the matrices and remove zeros
        var_cloud = self.var_matrix[np.triu_indices_from(self.var_matrix, k=1)].flatten()
        dist_list = self.dist_matrix[np.triu_indices_from(self.dist_matrix, k=1)].flatten()
        theta_list = self.theta_matrix[np.triu_indices_from(self.theta_matrix, k=1)].flatten()

        # Filter by azimuth
        mask = np.abs(theta_list - azimuth_rad) <= np.radians(tol)
        var_cloud = var_cloud[mask]
        dist_list = dist_list[mask]

        return self._plot_isotropic(dist_list, var_cloud, bins=bins, cloud=cloud, ax=ax)


if __name__ == "__main__":
    print("Module test!")