# Copyright (c) 2026 Jintao Li. 
# Zhejiang University (ZJU).
# 
# Licensed under the MIT License.


import numpy as np
from vispy.scene.widgets import Widget
from vispy.visuals.axis import AxisVisual, Ticker, _get_ticks_talbot


class ScaleTicker(Ticker):

    def __init__(
        self,
        axis,
        transform: callable = lambda x: f"{x:g}",
        unit: str = '',
        scale=1.0,
        **kwargs,
    ):
        super().__init__(axis, **kwargs)
        self.tick_transform = transform
        self.unit = unit
        self._scale = scale

    def update_scale(self, scale):
        self._scale = scale

    def _get_tick_frac_labels(self):
        """Get the major ticks, minor ticks, and major labels"""
        minor_num = 4  # number of minor ticks per major division
        if (self.axis.scale_type == 'linear'):
            domain = self.axis.domain
            domain = (domain[0] * self._scale, domain[1] * self._scale
                      )  # <- Core modification
            if domain[1] < domain[0]:
                flip = True
                domain = domain[::-1]
            else:
                flip = False
            offset = domain[0]
            scale = domain[1] - domain[0]

            transforms = self.axis.transforms
            length = self.axis.pos[1] - self.axis.pos[0]  # in logical coords
            n_inches = np.sqrt(np.sum(length**2)) / transforms.dpi

            major = _get_ticks_talbot(domain[0], domain[1], n_inches, 2)

            # labels = ['%g' % x for x in major]
            majstep = major[1] - major[0]
            minor = []
            minstep = majstep / (minor_num + 1)
            minstart = 0 if self.axis._stop_at_major[0] else -1
            minstop = -1 if self.axis._stop_at_major[1] else 0
            for i in range(minstart, len(major) + minstop):
                maj = major[0] + i * majstep
                minor.extend(
                    np.linspace(
                        maj + minstep,
                        maj + majstep - minstep,
                        minor_num,
                    ))
            major_frac = major - offset
            minor_frac = np.array(minor) - offset
            if scale != 0:  # maybe something better to do here?
                major_frac /= scale
                minor_frac /= scale
            use_mask = (major_frac > -0.0001) & (major_frac < 1.0001)
            major_frac = major_frac[use_mask]
            labels = [
                self.tick_transform(l) + self.unit
                for li, l in enumerate(major) if use_mask[li]
            ]
            minor_frac = minor_frac[(minor_frac > -0.0001)
                                    & (minor_frac < 1.0001)]
            # Flip ticks coordinates if necessary :
            if flip:
                major_frac = 1 - major_frac
                minor_frac = 1 - minor_frac
        elif self.axis.scale_type == 'logarithmic':
            return NotImplementedError
        elif self.axis.scale_type == 'power':
            return NotImplementedError
        return major_frac, minor_frac, labels


class ScaleAxisVisual(AxisVisual):

    def __init__(
        self,
        transform: callable = lambda x: f"{x:g}",
        unit: str = '',
        text_color='w',
        axis_color=(1, 1, 1), 
        tick_color=(0.7, 0.7, 0.7), 
        **kwargs,
    ):
        super().__init__(text_color=text_color, axis_color=axis_color, tick_color=tick_color, **kwargs)
        self.ticker = ScaleTicker(self, transform, unit)
        self._need_update = True


class AxisWidget(Widget):
    """Widget containing an axis

    Parameters
    ----------
    orientation : str
        Orientation of the axis, 'left' or 'bottom'.
    **kwargs : dict
        Keyword arguments to pass to AxisVisualLite.
    """

    def __init__(
        self,
        orientation='left',
        transform: callable = lambda x: f"{x:g}",
        unit: str = '',
        text_color='w',
        axis_color=(1, 1, 1), 
        tick_color=(0.7, 0.7, 0.7), 
        **kwargs,
    ):
        if 'tick_direction' not in kwargs:
            tickdir = {
                'left': (-1, 0),
                'right': (1, 0),
                'bottom': (0, 1),
                'top': (0, -1)
            }[orientation]
            kwargs['tick_direction'] = tickdir
        self.axis = ScaleAxisVisual(transform=transform, unit=unit, text_color=text_color, axis_color=axis_color, tick_color=tick_color, **kwargs)
        self.orientation = orientation
        self._linked_view = None
        self._last_domain = None
        Widget.__init__(self)
        self.add_subvisual(self.axis)

    def update_scale(self, scale):
        self.axis.ticker.update_scale(scale)
        self.axis._update_subvisuals()

    def on_resize(self, event):
        """Resize event handler

        Parameters
        ----------
        event : instance of Event
            The event.
        """
        self._update_axis()

    def force_update(self):
        self.axis._update_subvisuals()

    def _update_axis(self):
        self.axis.pos = self._axis_ends()

    def _axis_ends(self):
        r = self.rect
        if self.orientation == 'left':
            return np.array([[r.right, r.top], [r.right, r.bottom]])
        elif self.orientation == 'bottom':
            return np.array([[r.left, r.bottom], [r.right, r.bottom]])
        elif self.orientation == 'right':
            return np.array([[r.left, r.top], [r.left, r.bottom]])
        elif self.orientation == 'top':
            return np.array([[r.left, r.top], [r.right, r.top]])
        else:
            raise RuntimeError('Orientation %s not supported.' %
                               self.orientation)

    def link_view(self, view):
        """Link this axis to a ViewBox

        This makes it so that the axis's domain always matches the
        visible range in the ViewBox.

        Parameters
        ----------
        view : instance of ViewBox
            The ViewBox to link.
        """
        if view is self._linked_view:
            return
        if self._linked_view is not None:
            self._linked_view.scene.transform.changed.disconnect(self._view_changed)
        self._linked_view = view
        view.scene.transform.changed.connect(self._view_changed)
        self._view_changed()

    def _view_changed(self, event=None):
        """Linked view transform has changed; update ticks."""
        tr = self.node_transform(self._linked_view.scene)
        p1, p2 = tr.map(self._axis_ends())
        if self.orientation in ('left', 'right'):
            new_domain = (p1[1], p2[1])  # y
        else:
            new_domain = (p1[0], p2[0])  # x

        if not self._domain_changed(new_domain):
            return

        self._last_domain = new_domain
        self.axis.domain = new_domain

    def _domain_changed(self, new_domain, tol=1e-9):
        if self._last_domain is None:
            return True
        return not np.allclose(self._last_domain, new_domain, atol=tol)
