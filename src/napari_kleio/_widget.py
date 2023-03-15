"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

import numpy as np

from magicgui import magic_factory
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget

from napari.layers.labels._labels_utils import (
    indices_in_shape,
    interpolate_coordinates,
    sphere_indices,
)

from napari.layers.labels.labels import _coerce_indices_for_vectorization

from kleio.stores import VersionedFSStore, ZarrIndexStore

import zarr

if TYPE_CHECKING:
    import napari

def ZARR_PAINT(self, coord, new_label, refresh=True):
    """Paint over existing labels with a new label, using the selected
        brush shape and size, either only on the visible slice or in all
        n dimensions.
        Parameters
        ----------
        coord : sequence of int
            Position of mouse cursor in image coordinates.
        new_label : int
            Value of the new label to be filled in.
        refresh : bool
            Whether to refresh view slice or not. Set to False to batch paint
            calls.
        """
    shape = self.data.shape
    dims_to_paint = sorted(
        self._slice_input.order[-self.n_edit_dimensions :]
    )
    dims_not_painted = sorted(
        self._slice_input.order[: -self.n_edit_dimensions]
    )
    paint_scale = np.array(
        [self.scale[i] for i in dims_to_paint], dtype=float
    )

    slice_coord = [int(np.round(c)) for c in coord]
    if self.n_edit_dimensions < self.ndim:
        coord_paint = [coord[i] for i in dims_to_paint]
        shape = [shape[i] for i in dims_to_paint]
    else:
        coord_paint = coord
        
    # Ensure circle doesn't have spurious point
    # on edge by keeping radius as ##.5
    radius = np.floor(self.brush_size / 2) + 0.5
    mask_indices = sphere_indices(radius, tuple(paint_scale))
    
    mask_indices = mask_indices + np.round(np.array(coord_paint)).astype(
        int
    )
    
    # discard candidate coordinates that are out of bounds
    mask_indices = indices_in_shape(mask_indices, shape)
    
    # Transfer valid coordinates to slice_coord,
    # or expand coordinate if 3rd dim in 2D image
    slice_coord_temp = list(mask_indices.T)
    if self.n_edit_dimensions < self.ndim:
        for j, i in enumerate(dims_to_paint):
            slice_coord[i] = slice_coord_temp[j]
        for i in dims_not_painted:
            slice_coord[i] = slice_coord[i] * np.ones(
                mask_indices.shape[0], dtype=int
            )
    else:
        slice_coord = slice_coord_temp

    slice_coord = _coerce_indices_for_vectorization(self.data, slice_coord)

    # slice coord is a tuple of coordinate arrays per dimension
    # subset it if we want to only paint into background/only erase
    # current label
    if self.preserve_labels:
        if new_label == self._background_label:
            keep_coords = self.data[slice_coord] == self.selected_label
        else:
            keep_coords = self.data[slice_coord] == self._background_label
            slice_coord = tuple(sc[keep_coords] for sc in slice_coord)

    self.data_setitem(slice_coord, new_label, refresh)

    # TODO write new coords to the zarr

class KleioWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        
        create_btn = QPushButton("Create")
        create_btn.clicked.connect(self._on_create)

        # open
        open_btn = QPushButton("Open")
        open_btn.clicked.connect(self._on_open)
        
        # commit
        commit_btn = QPushButton("Commit")
        commit_btn.clicked.connect(self._on_commit)

        # checkout
        checkout_btn = QPushButton("Checkout")
        checkout_btn.clicked.connect(self._on_checkout)
        
        # create branch
        create_branch_btn = QPushButton("Create Branch")
        create_branch_btn.clicked.connect(self._on_create_branch)

        # TODO text box for index_path
        self.index_path = "/tmp/test_zarr/index"
        
        # TODO text box for raw_path
        self.raw_path = "/tmp/test_zarr/raw"

        # TODO text box for dataset
        self.dataset = "test_dataset"
        
        # TODO git history with selectable versions
        
        # TODO select labels layer (init self.annotation_layer)
        self.annotation_layer = self.viewer.layers["Labels"]

        # TODO override paint to record to zarr https://github.com/napari/napari/blob/45044be5c04f79a7a59e2ce229e8ee97248ef4fc/napari/layers/labels/labels.py#L1270
        
        self.setLayout(QHBoxLayout())
        self.layout().addWidget(create_btn)
        self.layout().addWidget(open_btn)
        self.layout().addWidget(commit_btn)
        self.layout().addWidget(checkout_btn)
        self.layout().addWidget(create_branch_btn)    
        
    def toggle_paint_mode(self):
        napari.layers.labels.Labels.paint = ZARR_PAINT
        
    def _on_create(self):
        # Save a new
        index_store = ZarrIndexStore(self.index_path)
        store = VersionedFSStore(index_store, self.raw_path)

        z = zarr.open(store, mode="a")
        z.create_dataset(self.dataset, shape=self.annotation_layer.data.shape, dtype=self.annotation_layer.data.dtype)
        z[self.dataset] = self.annotation_layer.data

        # Commit
        store.vc.commit("Commiting!")
        

    def _on_open(self):
        print("napari has", len(self.viewer.layers), "layers")

    def _on_commit(self):
        index_store = ZarrIndexStore(self.index_path)
        store = VersionedFSStore(index_store, self.raw_path)

        z = zarr.open(store, mode="a")

        z[self.dataset] = self.annotation_layer.data

        # Commit
        store.vc.commit("Commiting!")


    def _on_checkout(self):
        print("napari has", len(self.viewer.layers), "layers")

    def _on_create_branch(self):
        print("napari has", len(self.viewer.layers), "layers")

if __name__ == "__main__":
    
    import napari

    viewer = napari.Viewer()    
    
    viewer.add_image(np.ones((100, 100, 100)))
    viewer.add_layer(napari.layers.Labels(np.zeros((100, 100, 100), dtype=np.uint16), name="Labels"))

    widget = KleioWidget(viewer)

    index_store = ZarrIndexStore(widget.index_path)
    store = VersionedFSStore(index_store, widget.raw_path)

    z = zarr.open(store, mode="a")

    viewer.add_image(z[widget.dataset][:])
    
    viewer.window.add_dock_widget(widget)


