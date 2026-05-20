import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulation.febio_parser import FEBio2Dolfinx

from tests.conftest import _require_single_rank


def test_region_index_map_exposes_labels_and_cells(tmp_path):
    _require_single_rank()
    feb_content = """<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <Module type="solid"/>
  <Material>
    <material id="1" name="BoneMaterial" type="isotropic elastic">
      <density>1</density>
      <E>12000</E>
      <v>0.3</v>
    </material>
    <material id="2" name="CartMaterial" type="isotropic elastic">
      <density>1</density>
      <E>500</E>
      <v>0.45</v>
    </material>
  </Material>
  <Mesh>
    <Nodes name="AllNodes">
      <node id="1">0,0,0</node>
      <node id="2">1,0,0</node>
      <node id="3">0,1,0</node>
      <node id="4">0,0,1</node>
      <node id="5">1,1,0</node>
      <node id="6">1,0,1</node>
    </Nodes>
    <Elements type="tet4" name="PelvisBone">
      <elem id="1">1,2,3,4</elem>
    </Elements>
    <Elements type="tet4" name="LeftCartilage">
      <elem id="2">2,3,5,6</elem>
    </Elements>
  </Mesh>
  <MeshDomains>
    <SolidDomain name="PelvisBone" mat="BoneMaterial"/>
    <SolidDomain name="LeftCartilage" mat="CartMaterial"/>
  </MeshDomains>
</febio_spec>
"""
    feb_path = tmp_path / "sample.feb"
    feb_path.write_text(feb_content, encoding="ISO-8859-1")

    model = FEBio2Dolfinx(str(feb_path))

    assert list(model.region_labels) == ["PelvisBone", "LeftCartilage"]
    mapping = model.region_index_map

    assert set(mapping.keys()) == {"PelvisBone", "LeftCartilage"}
    cell_indices = np.asarray(model.cell_tag.indices, dtype=np.int32)
    # Cells are contiguous and cover all entries once
    all_indices = np.sort(np.concatenate([mapping["PelvisBone"], mapping["LeftCartilage"]]))
    assert np.array_equal(all_indices, np.sort(cell_indices))

    bone_cells = np.asarray(mapping["PelvisBone"], dtype=np.int32)
    cart_cells = np.asarray(mapping["LeftCartilage"], dtype=np.int32)
    assert bone_cells.size > 0
    assert cart_cells.size > 0

    bone_tag = model.region_label_map["PelvisBone"]
    cart_tag = model.region_label_map["LeftCartilage"]

    assert np.array_equal(np.sort(bone_cells), np.sort(model.cell_tag.find(bone_tag)))
    assert np.array_equal(np.sort(cart_cells), np.sort(model.cell_tag.find(cart_tag)))

    assert np.all(model.region_ids[bone_cells] == bone_tag)
    assert np.all(model.region_ids[cart_cells] == cart_tag)


def test_febio_missing_nodes_raises(tmp_path):
    """Parser should raise when <Nodes> are missing."""
    from pytest import raises
    feb_content = """<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>
<febio_spec version=\"4.0\">
  <Module type=\"solid\"/>
  <Mesh>
    <Elements type=\"tet4\" name=\"Region\">
      <elem id=\"1\">1,2,3,4</elem>
    </Elements>
  </Mesh>
</febio_spec>
"""
    feb_path = tmp_path / "bad_missing_nodes.feb"
    feb_path.write_text(feb_content, encoding="ISO-8859-1")
    with raises(ValueError, match="No <node> elements"):
        _ = FEBio2Dolfinx(str(feb_path))


def test_febio_unsupported_element_type_raises(tmp_path):
    """Unsupported element types should raise ValueError."""
    from pytest import raises
    feb_content = """<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>
<febio_spec version=\"4.0\">
  <Module type=\"solid\"/>
  <Mesh>
    <Nodes name=\"AllNodes\">
      <node id=\"1\">0,0,0</node>
      <node id=\"2\">1,0,0</node>
      <node id=\"3\">0,1,0</node>
      <node id=\"4\">0,0,1</node>
      <node id=\"5\">1,1,0</node>
      <node id=\"6\">1,0,1</node>
      <node id=\"7\">0,1,1</node>
      <node id=\"8\">1,1,1</node>
    </Nodes>
    <Elements type=\"hex8\" name=\"Block\">
      <elem id=\"1\">1,2,3,4,5,6,7,8</elem>
    </Elements>
  </Mesh>
</febio_spec>
"""
    feb_path = tmp_path / "bad_element_type.feb"
    feb_path.write_text(feb_content, encoding="ISO-8859-1")
    with raises(ValueError, match="Unsupported element type"):
        _ = FEBio2Dolfinx(str(feb_path))
