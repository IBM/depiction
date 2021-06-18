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
        super(FeatureAttributionExplanation, self).__init__()
        self.attributions = feat_attr
        self.data_type = data_type

    def normalize(self, vmin=None, vmax=None):
        if vmin is None:
            vmin = np.min(self.attributions)
        if vmax is None:
            vmax = np.max(self.attributions)
        self.attributions = (self.attributions - vmin)/(vmin-vmax)

    def _visualize_tabular(self, *, feature_names=None, fig_axes: tuple = None, plot_mode: str='heatmap',
                           show=False, **kwargs):
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


    def visualize(self, *args, **kwargs):
        if self.data_type == DataType.TABULAR:
            fig, ax = self._visualize_tabular(*args, **kwargs)
        elif self.data_type == DataType.IMAGE:
            fig, ax = self._visualize_image(*args, **kwargs)
        elif self.data_type == DataType.TEXT:
            fig, ax = self._visualize_text(*args, **kwargs)
        else:
            raise RuntimeError('There might have been an error in setting the data type!')
        return fig, ax
