import numpy as np
from depiction.explanations.base import BaseExplanation
from depiction.core import DataType
from matplotlib import pyplot as plt
import seaborn as sns
from copy import deepcopy
from highlight_text import HighlightText
from matplotlib import cm
from matplotlib.colors import rgb2hex


class FeatureAttributionExplanation(BaseExplanation):

    def __init__(self, feat_attr: np.ndarray, data_type: DataType):
        """
        Constructor

        Args:
            feat_attr: np.array containing the attributions
            data_type: data type of the explained model
        """
        super(FeatureAttributionExplanation, self).__init__()
        self.attributions = feat_attr
        self.data_type = data_type


    def normalize(self, vmin=None, vmax=None, clip=False):
        """
        Normalize the attributions to the range [0, 1] given a min and a max value.

        Args:
            vmin: min value to use for normalization
            vmax: max value to use for normalization
            clip: if True, clip attributions before normalization
        """
        if vmin is None:
            vmin = np.min(self.attributions)
        if vmax is None:
            vmax = np.max(self.attributions)
        if clip:
            self.attributions = np.clip(self.attributions, a_min=vmin, a_max=vmax)
        self.attributions = (self.attributions - vmin)/(vmin-vmax)

    def _visualize_tabular(self, *, feature_names=None, fig_axes: tuple = None, plot_mode: str='heatmap',
                           show=False, **kwargs):
        """
        Routine to visualize feature attributions for tabular data

        Args:
            feature_names: iterable containing the names of the features. Default: None.
            fig_axes: tuple containing plt.Figure and plt.Axis instances where to plot the attributions
            plot_mode: string specifying how to plot the attributions. Available modes: 'barplot', 'heatmap'
            show: if True, plot the generated figure.
            kwargs: additional arguments to pass to the plotting functions. refer to seaborn for more details.
        Returns:
            fig: plt.Figure
            ax: plt.Axis
        """
        PLOT_MODES = {'barplot', 'heatmap'}
        if plot_mode not in PLOT_MODES:
            raise RuntimeError('Mode "{}" not valid. Valid modes are: {}'.format(plot_mode, PLOT_MODES))

        if feature_names is None:
            feature_names = ['Feat{}'.format(i) for i in range(self.attributions.shape[-1])]
        else:
            if len(feature_names) != self.attributions.shape[-1]:
                raise RuntimeError('Mismatch in dim between attributions and provided "feature_names"!')

        if fig_axes is None:
            fig, ax = plt.subplots()
        else:
            fig, ax  = fig_axes

        kwargs.update({'ax': ax})

        if plot_mode == 'barplot':
            sns.barplot(x=feature_names, y=self.attributions, **kwargs)
        else:
            sns.heatmap(data=self.attributions, ax=ax, **kwargs)

        ax.set_xticklabels(feature_names)

        if show:
            fig.show()

        return fig, ax

    def _visualize_image(self, *, image: np.ndarray=None, fig_axes: tuple = None, show=False, **kwargs):
        """
        Routine to visualize feature attributions for tabular data

        Args:
            image: reference image to plot alongside the attributions
            fig_axes: tuple containing plt.Figure and plt.Axis instances where to plot the attributions
            show: if True, plot the generated figure.
            kwargs: additional arguments to pass to the plt.imshow
        Returns:
            fig: plt.Figure
            ax: plt.Axis
        """
        image = deepcopy(image)
        # check dim consistency
        if self.attributions.shape != image.shape:
            raise RuntimeError('Attributions should have the same size as input image')

        if self.attributions.shape != 3:
            raise RuntimeError('We expect the image and the attributions to have only 3 dimensions for visualization:'
                               'height, width and channels. That is we can visualize only one sample at a time!')

        if self.attributions.shape[-1] > 3:
            self.attributions = np.transpose(self.attributions, 0, 1)
            self.attributions = np.transpose(self.attributions, 1, 2)

        attributions = np.sum(self.attributions, axis=2, keepdims=True)

        if image.shape[-1] > 3:
            image = np.transpose(image, 0, 1)
            image = np.transpose(image, 1, 2)

        if fig_axes is None:
            fig, ax = plt.subplots()
        else:
            fig, ax  = fig_axes

        keyword_args = {
            'alpha': 0.5
        }
        keyword_args.update(kwargs)
        ax.imshow(self.attributions, **kwargs)
        keyword_args['alpha'] = 1.0 - keyword_args['alpha']
        keyword_args.pop('cmap', None)
        ax.imshow(image, **kwargs)

        if show:
            fig.show()

        return fig, ax

    def _visualize_text(self, *, tokens, delimiter=' ', fig_axes: tuple = None, n_colors = 20, show=False, hl_chars=('<', '>'), **kwargs):
        """
        Routine to visualize feature attributions for tabular data

        Args:
            tokens: tokenized string for which the attributions should be visualized.
            delimiter: delimiter to use to recompose the string. Default: ' '.
            fig_axes: tuple containing plt.Figure and plt.Axis instances where to plot the attributions
            n_colors: number of bins to use to discretize the attributions for visualization
            show: if True, plot the generated figure.
            hl_chars: tuple containing the delimiters for highlighting the text according the attributions. Please refer to the
                        highlight_text documentation for further details. NOTE: characters not included in the vocabulary should
                        be used.
        Returns:
            fig: plt.Figure
            ax: plt.Axis
        """
        if len(tokens) != self.attributions.shape[-1]:
            raise UserWarning('The number of tokens and attributions must correspond!')

        cmap = kwargs.pop('cmap', 'bwr')
        cmap = cm.get_cmap(cmap, n_colors)
        list_cmap = [rgb2hex(cmap(i)) for i in range(cmap.N)]
        attributions = np.digitize(self.attributions,
                                   bins=np.linspace(np.min(self.attributions), np.max(self.attributions), num=n_colors, endpoint=True), right=False) - 1

        if fig_axes is None:
            fig, ax = plt.subplots()
        else:
            fig, ax  = fig_axes

        s = []
        colors = []
        for i in range(len(tokens)):
            s.append(hl_chars[0] + tokens[i] + hl_chars[1])
            colors.append({"bbox": {"facecolor": list_cmap[attributions[i]], "linewidth": 0.2, "pad": 0.2, "edgecolor": list_cmap[attributions[i]]}})
        s = delimiter.join(s)
        HighlightText(0.5, 0.5, s=s, ha='center', va='center', highlight_textprops=colors, ax=ax, delim=hl_chars,
                      fontsize=24)
        ax.grid(False)
        ax.axis('off')

        if show:
            fig.show()

        return fig, ax


    def visualize_help(self):
        if self.data_type == DataType.TABULAR:
            print(self._visualize_tabular.__doc__)
        elif self.data_type == DataType.IMAGE:
            print(self._visualize_image.__doc__)
        elif self.data_type == DataType.TEXT:
            print(self._visualize_text.__doc__)
        else:
            raise RuntimeError('There might have been an error in setting the data type!')


    def visualize(self, *args, **kwargs):
        """
        Interface method for plotting
        """
        if self.data_type == DataType.TABULAR:
            fig, ax = self._visualize_tabular(*args, **kwargs)
        elif self.data_type == DataType.IMAGE:
            fig, ax = self._visualize_image(*args, **kwargs)
        elif self.data_type == DataType.TEXT:
            fig, ax = self._visualize_text(*args, **kwargs)
        else:
            raise RuntimeError('There might have been an error in setting the data type!')
        return fig, ax
